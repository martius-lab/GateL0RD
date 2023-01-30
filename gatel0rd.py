import torch
import torch.nn as nn


def ReTanh(x):
    """
    ReTanh function applied on tensor x
    """
    return x.tanh().clamp(min=0, max=1)


class HeavisideST(torch.autograd.Function):
    """
    Heaviside activation function with straight through estimator
    """

    @staticmethod
    def forward(ctx, input):
        return torch.ceil(input).clamp(min=0, max=1)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class GateL0RDCell(nn.Module):
    """

    RNNCell of GateL0RD

    One GateL0RD cell uses three subnetworks:
    1. a recommendation network r, which proposes a new candidate latent state
    2. a gating network g, which determines how the latent state is updated
    3. output functions (p & o), which computes the output based on the updated latent state and the input.

    The forward pass computes the following from input x_t and previous latent state h_{t-1}:
    - s_t \sim \mathcal{N}(g(x_t, h_{t-1}), \Sigma)
    - \Lambda(s_t) = max(0, \tanh(s_t))
    - h_t =  \Lambda(s_t) \odot r(x_t, h_{t-1}) + (1 - \Lambda(s_t)) \odot h_{t-1}
    - y_t = p(x_t, h_t) \odot p(x_t, h_t)

    """

    def __init__(self, input_size, hidden_size, reg_lambda, output_size=-1, num_layers_internal=1, gate_noise_level=0.1,
                 device=None):
        """
        GateL0RD cell
        :param input_size: The number of expected features in the cell input x
        :param hidden_size: The number of features in the latent state h
        :param reg_lambda: Hyperparameter controlling the sparsity of latent state changes
        :param output_size: The number of expected features for the cell output y (Default: same as hidden size)
        :param num_layers_internal: Number of layers used in the g - and r-subnetworks
        :param gate_noise_level: Standard deviation of normal distributed gate noise for stochastic gates (\Sigma)
        :param device: torch.device to use for creating tensors.
        """

        super(GateL0RDCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        if output_size == -1:
            output_size = hidden_size
        self.output_size = output_size

        input_size_gates = input_size + hidden_size

        # Create g-network:
        temp_gating = nn.ModuleList([])
        in_dim_g = input_size_gates
        for gl in range(num_layers_internal):
            gl_factor = pow(2, (num_layers_internal - gl - 1))
            out_dim_g = gl_factor * hidden_size
            temp_gating.append(nn.Linear(in_dim_g, out_dim_g))

            if gl < (num_layers_internal - 1):
                temp_gating.append(nn.Tanh())
            in_dim_g = out_dim_g
        self.input_gates = nn.Sequential(*temp_gating)

        # Create r-network:
        temp_r_function = nn.ModuleList([])
        in_dim_r = input_size_gates
        for rl in range(num_layers_internal):
            rl_factor = pow(2, (num_layers_internal - rl - 1))
            out_dim_r = rl_factor * hidden_size
            temp_r_function.append(nn.Linear(in_dim_r, out_dim_r))
            temp_r_function.append(nn.Tanh())
            in_dim_r = out_dim_r
        self.r_function = nn.Sequential(*temp_r_function)

        # Create output function p * o:

        # Create p-network:
        temp_output_function = nn.ModuleList([])
        temp_output_function.append(nn.Linear(input_size_gates, output_size))
        temp_output_function.append(nn.Tanh())
        self.output_function = nn.Sequential(*temp_output_function)

        # Create o-network
        temp_outputgate = nn.ModuleList([])
        temp_outputgate.append(nn.Linear(input_size_gates, output_size))
        temp_outputgate.append(nn.Sigmoid())
        self.output_gates = nn.Sequential(*temp_outputgate)

        assert gate_noise_level >= 0, "Need a positive standard deviation as the gate noise"
        self.gate_noise_level = gate_noise_level

        # Gate regularization
        self.reg_lambda = reg_lambda
        self.gate_reg = HeavisideST.apply

        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device

    def forward(self, x_t, h_tminus1=None):
        """
        Forward pass one step, i.e. pass through g-, r-, p- and o-subnetwork
        :param x_t: tensor of cell inputs
        :param h_tminus1: tensor of last latent state (Default: initialized by zeros)
        :return: rnn output y_t, hidden states h_t, tensor of regularized gatings \Theta(\Lambda_t)
        """
        assert len(x_t.shape) == 2, "Wrong input dimensionality of x_t in GateL0RDCell: " + str(x_t.shape)
        batch_size, layer_input_size = x_t.size()

        if h_tminus1 is None:
            h_tminus1 = torch.zeros((batch_size, self.hidden_size), device=self.device)
        else:
            assert len(h_tminus1.shape) == 2, "Wrong input dimensionality of h_tminus1 in GateL0RDCell: " + str(h_tminus1.shape)
            assert h_tminus1.shape[1] == self.hidden_size

        # Input to g and r-network is the current input plus the last latent state
        gr_input = torch.cat((x_t, h_tminus1), 1)

        '''
        G- NETWORK
        '''
        i_t = self.input_gates(gr_input)
        if self.training:
            gate_noise = torch.randn(size=(batch_size, self.hidden_size), device=self.device) * self.gate_noise_level
        else:
            # Gate noise is zero
            gate_noise = torch.zeros((batch_size, self.hidden_size), device=self.device)

        # Stochastic input gate activation
        Lambda_t = ReTanh(i_t - gate_noise)
        Theta_t = self.gate_reg(Lambda_t)

        '''
        R-Network
        '''
        h_hat_t = self.r_function(gr_input)

        '''
        New latent state
        '''
        h_t = Lambda_t * h_hat_t + (1.0 - Lambda_t) * h_tminus1

        '''
        Output function :
        '''
        xh_t = torch.cat((x_t, h_t), 1)
        y_hat_t = self.output_function(xh_t)

        # Output is computed as p(x_t, h_t) * o(x_t, h_t)
        o_lt = self.output_gates(xh_t)
        y_t = y_hat_t * o_lt

        return y_t, h_t, Theta_t

    def loss(self, loss_task, Theta):
        """
        GateL0RD loss function
        :param loss_task: Computed task-based loss, e.g. MSE for regression or cross-entropy for classification
        :param Theta: Regularized gate activation
        :return: lambda-weighted sum of the two losses
        """
        assert Theta is not None, 'Provide tensor of regularized gates (Theta) for loss computation.'
        gate_loss = torch.mean(Theta)
        return loss_task + self.reg_lambda * gate_loss


class GateL0RD(torch.nn.Module):
    """

    RNN implementation of GateL0RD

    """

    def __init__(self, input_size, hidden_size, reg_lambda, output_size=-1, num_layers_internal=1,
                 h_init_net=False, h_init_net_layers=3, gate_noise_level=0.1, batch_first=False, device=None):
        """
        GateL0RD RNN
        :param input_size: The number of expected features in the cell input x
        :param hidden_size: The number of features in the latent state h
        :param reg_lambda: Hyperparameter controlling the sparsity of latent state changes
        :param output_size: The number of expected features for the cell output y (Default: same as hidden size)
        :param h_init_net: If true, then use a feed-forward network to learn to initialize the hidden state based on the
        first input (Default: False)
        :param h_init_net_layers: How many layers will be used to initialize the hidden state from the input. Layer
        number l has 2^l*hidden_size features. (Default: 3 layers)
        :param num_layers_internal: Number of layers used in the g - and r-subnetworks
        :param gate_noise_level: Standard deviation of normal distributed gate noise for stochastic gates (\Sigma)
        :param batch_first: If True, then the input and output tensors are provided as (batch, seq, feature) instead of
        (seq, batch, feature). Note that this does not apply to latent states or gates. Default: False
        :param device: torch.device to use for creating tensors.
        """

        super(GateL0RD, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        if output_size == -1:
            output_size = hidden_size
        self.output_size = output_size

        self.cell = GateL0RDCell(input_size=input_size, output_size=output_size, hidden_size=hidden_size,
                                 num_layers_internal=num_layers_internal, gate_noise_level=gate_noise_level,
                                 reg_lambda=reg_lambda, device=device)

        self.use_h_init_net = h_init_net
        if h_init_net:
            self.f_init = self.__create_f_init(f_init_layers=h_init_net_layers, input_dim=input_size,
                                               latent_dim=hidden_size)

        self.last_Thetas = None

        self.batch_first = batch_first

        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device

    @staticmethod
    def __create_f_init(f_init_layers, input_dim, latent_dim):
        input_dim_warm_up = input_dim
        warm_up_net = nn.ModuleList([])
        for w in range(f_init_layers):
            w_factor = pow(2, (f_init_layers - w - 1))
            warm_up_net.append(nn.Linear(input_dim_warm_up, w_factor * latent_dim))
            warm_up_net.append(nn.Tanh())
            input_dim_warm_up = w_factor * latent_dim
        return nn.Sequential(*warm_up_net)

    def __forward_one_step(self, x_t, h_tminus1):
        return self.cell.forward(x_t, h_tminus1)

    def forward(self, x, h_init=None, h_sequence=False):
        """
        Forward pass for sequence data
        :param x: tensor of sequence of input batches with shape (seq, batch, feature)
        (or (batch, seq, feature) for batch_first=True)
        :param h_init: tensor of initial latent state with shape (1, batch, feature). If None it is initialized by a
        feed-forward network based on x_0 (h_init_net=True) or set to zero (h_init_net=False).
        :param h_sequence: If True outputs sequence of latent states, else only last latent state (Default:False)
        :return:    - rnn output y with shape (seq, batch, feature) (or (batch, seq, feature) for batch_first=True),
                    - latent state h of shape (1, batch, feature) (or (seq, batch, feature) for output_h_sequence=True),
                    - regularized gate activations (\Theta(\Lambda(s))) with shape (seq, batch, feature)
        """

        assert len(x.shape) == 3, "Input must have 3 dimensions, got " + str(len(x.shape))

        if self.batch_first:
            x = x.permute(1, 0, 2)

        S, B, D = x.shape

        assert D == self.input_size, "Expected input of shape (*, *, " + str(self.input_size) + "), got " + str(x.shape)

        if h_init is None:
            if self.use_h_init_net:
                h_init = self.f_init(x[0, :, :]).unsqueeze(0)
            else:
                h_init = torch.zeros((1, B, self.hidden_size), device=self.device)
        else:
            h_shape = h_init.shape
            assert len(h_shape) == 3 and h_shape[0] == 1 and h_shape[1] == B and h_shape[2] == self.hidden_size, \
                "Expected latent state of shape (1, " + str(B) + ", " + str(self.hidden_size) + "), got " + str(h_shape)

        h_t = h_init[0, :, :]
        list_ys = []
        list_hs = []
        list_Thetas = []
        for t in range(S):
            x_t = x[t, :, :]
            y_t, h_t, Theta_t = self.__forward_one_step(x_t=x_t, h_tminus1=h_t)
            list_ys.append(y_t)
            list_hs.append(h_t)
            list_Thetas.append(Theta_t)

        ys = torch.stack(list_ys)
        hs = torch.stack(list_hs)
        Thetas = torch.stack(list_Thetas)
        h_output = h_t.unsqueeze(0)

        if self.batch_first:
            ys = ys.permute(1, 0, 2)

        self.last_Thetas = Thetas

        if h_sequence:
            h_output = hs

        return ys, h_output, Thetas

    def loss(self, loss_task, Theta=None):
        """
        GateL0RD loss function
        :param loss_task: Computed task-based loss, e.g. MSE for regression or cross-entropy for classification
        :param Theta: Regularized gate activation, Default: Gate activation from last forward-call
        :return: lambda-weighted sum of the two losses
        """
        if Theta is None:
            assert self.last_Thetas is not None, "forward() needs to be called before loss computation."
            return self.cell.loss(loss_task, self.last_Thetas)
        return self.cell.loss(loss_task, Theta)
