import torch
from torch import nn
from torch.distributions import Normal

from humanoid.algo.ppo.actor_critic import ActorCritic


def get_conv1d_output_size(input_size, kernel_size, stride):
    return (input_size - kernel_size) // stride + 1


class LongShortActor(nn.Module):
    def __init__(
            self,
            num_actor_obs,
            num_actions,
            actor_hidden_dims=(256, 256, 256),
            conv1d_params=((32, 6, 3), (16, 4, 2)),
            activation=nn.ELU(),
            long_obs_length=100,
            short_obs_length=5,
    ):
        super(LongShortActor, self).__init__()

        self.single_obs_dim = num_actor_obs // long_obs_length
        self.long_obs_length = long_obs_length
        self.short_obs_length = short_obs_length

        actor_cnn_layers = []
        in_channels, in_shape = self.single_obs_dim, long_obs_length
        for out_channels, kernel_size, stride in conv1d_params:
            actor_cnn_layers.append(
                nn.Conv1d(in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride))
            actor_cnn_layers.append(activation)
            in_channels, in_shape = out_channels, get_conv1d_output_size(in_shape, kernel_size, stride)
        self.actor_cnn = nn.Sequential(*actor_cnn_layers)

        mlp_input_dim_a = self.short_obs_length * self.single_obs_dim + in_shape * conv1d_params[-1][0]
        actor_mlp_layers = []
        actor_mlp_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_mlp_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_mlp_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_mlp_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_mlp_layers.append(activation)
        self.actor_mlp = nn.Sequential(*actor_mlp_layers)

        print(f"cnn size: { sum(p.numel() for p in self.actor_cnn.parameters())}")
        print(f"mlp size: { sum(p.numel() for p in self.actor_mlp.parameters())}")

    def forward(self, observations):
        long_hist = observations.reshape(-1, self.long_obs_length, self.single_obs_dim)
        long_hist_embed = self.actor_cnn(long_hist.transpose(1, 2))
        short_hist = long_hist[:, -self.short_obs_length:]
        mlp_inputs = torch.cat([long_hist_embed.flatten(1), short_hist.flatten(1)], dim=1)
        return self.actor_mlp(mlp_inputs)

    def numel(self):
        return sum(p.numel() for p in self.parameters())


class LongShortActorCritic(ActorCritic):
    def __init__(self,
                 num_actor_obs,
                 num_critic_obs,
                 num_actions,
                 actor_hidden_dims=(256, 256, 256),
                 conv1d_params=((32, 6, 3), (16, 4, 2)),
                 critic_hidden_dims=(256, 256, 256),
                 init_noise_std=1.0,
                 activation=nn.ELU(),
                 long_obs_length=100,
                 short_obs_length=5,
                 **kwargs):
        super(ActorCritic, self).__init__()

        self.actor = LongShortActor(
            num_actor_obs, num_actions, actor_hidden_dims, conv1d_params,
            activation, long_obs_length, short_obs_length
        )

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor Long-Short: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def numel(self):
        return sum(p.numel() for p in self.parameters())

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("path", type=str, default=None)

    # args = parser.parse_args()

    obs_dim = 50

    model = LongShortActorCritic(
        obs_dim * 100,
        288 * 3,
        26,
        long_obs_length=100,
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(768, 256, 128),
    )
    # model.load_state_dict(torch.load(args.path, weights_only=True)["model_state_dict"])

    obs = torch.zeros(32, obs_dim * 100)
    model.update_distribution(obs)

    print(f"Total number of trainable parameters: {model.numel()}")

    x = model.actor(obs)
    print(x.shape)

    # onnx_path = args.path.replace('.pt', '.onnx')
    # inputs = torch.rand([1, obs.shape[-1]])
    # torch.onnx.export(model.actor, inputs, onnx_path)
