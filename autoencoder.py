import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        # TODO: Implement a CNN. Save the layers in the modules list.
        # The input shape is an image batch: (N, in_channels, H_in, W_in).
        # The output shape should be (N, out_channels, H_out, W_out).
        # You can assume H_in, W_in >= 64.
        # Architecture is up to you, but you should use at least 3 Conv layers.
        # You can use any Conv layer parameters, use pooling or only strides,
        # use any activation functions, use BN or Dropout, etc.
        # ====== YOUR CODE: ======
        modules.append(nn.Conv2d(in_channels, 128, 5, padding=2, stride=2))
        modules.append(nn.LeakyReLU(0.2))
        modules.append(nn.Conv2d(128, 256, 5, padding=2, stride=2))
        modules.append(nn.LeakyReLU(0.2))
        modules.append(nn.Conv2d(256, 512, 5, padding=2, stride=2))
        modules.append(nn.LeakyReLU(0.2))
        modules.append(nn.Conv2d(512, out_channels, 5, padding=2, stride=2))
        modules.append(nn.LeakyReLU(0.2))
        # ========================
        self.cnn = nn.Sequential(*modules)

    def forward(self, x):
        return self.cnn(x)


class DecoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        # TODO: Implement the "mirror" CNN of the encoder.
        # For example, instead of Conv layers use transposed convolutions,
        # instead of pooling do unpooling (if relevant) and so on.
        # You should have the same number of layers as in the Encoder,
        # and they should produce the same volumes, just in reverse order.
        # Output should be a batch of images, with same dimensions as the
        # inputs to the Encoder were.
        # ====== YOUR CODE: ======
        modules.append(nn.ConvTranspose2d(in_channels, 512, 5, padding=2, stride=2, output_padding=1))
        modules.append(nn.LeakyReLU(0.2))
        modules.append(nn.ConvTranspose2d(512, 256, 5, padding=2, stride=2, output_padding=1))
        modules.append(nn.LeakyReLU(0.2))
        modules.append(nn.ConvTranspose2d(256, 128, 5, padding=2, stride=2, output_padding=1))
        modules.append(nn.LeakyReLU(0.2))
        modules.append(nn.ConvTranspose2d(128, out_channels, 5, padding=2, stride=2, output_padding=1))

        # ========================
        self.cnn = nn.Sequential(*modules)

    def forward(self, h):
        # Tanh to scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(self.cnn(h))


class VAE(nn.Module):
    def __init__(self, features_encoder, features_decoder, in_size, z_dim):
        """
        :param features_encoder: Instance of an encoder the extracts features
        from an input.
        :param features_decoder: Instance of a decoder that reconstructs an
        input from it's features.
        :param in_size: The size of one input (without batch dimension).
        :param z_dim: The latent space dimension.
        """
        super().__init__()
        self.features_encoder = features_encoder
        self.features_decoder = features_decoder
        self.z_dim = z_dim

        self.features_shape, n_features = self._check_features(in_size)

        # TODO: Add parameters needed for encode() and decode().
        # ====== YOUR CODE: ======
        # raise NotImplementedError()
        self.encoded_to_mu = nn.Linear(n_features,z_dim)
        self.encoded_to_log_sigma2 = nn.Linear(n_features,z_dim)
        self.z_to_decoder = nn.Linear(z_dim,n_features)
        # ========================

    def _check_features(self, in_size):
        device = next(self.parameters()).device
        with torch.no_grad():
            # Make sure encoder and decoder are compatible
            x = torch.randn(1, *in_size, device=device)
            h = self.features_encoder(x)
            xr = self.features_decoder(h)
            assert xr.shape == x.shape
            # Return the shape and number of encoded features
            return h.shape[1:], torch.numel(h)//h.shape[0]

    def encode(self, x):
        # TODO: Sample a latent vector z given an input x.
        # 1. Use the features extracted from the input to obtain mu and
        # log_sigma2 (mean and log variance) of the posterior p(z|x).
        # 2. Apply the reparametrization trick.
        # ====== YOUR CODE: ======
        encoded = self.features_encoder(x).view(x.shape[0], -1)
        mu = self.encoded_to_mu(encoded).view(1, -1)
        log_sigma2 = self.encoded_to_log_sigma2(encoded).view(1, -1)
        std = log_sigma2.mul(0.5).exp_()
        z = mu + std * torch.randn(*mu.size(), device=std.device)
        # ========================

        return z, mu, log_sigma2

    def decode(self, z):
        # TODO: Convert a latent vector back into a reconstructed input.
        # 1. Convert latent to features.
        # 2. Apply features decoder.
        # ====== YOUR CODE: ======
        decoder_inp = self.z_to_decoder(z.view(-1,self.z_dim))
        decoder_inp = decoder_inp.view(-1, *self.features_shape)
        x_rec = self.features_decoder(decoder_inp)
        # ========================

        # Scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(x_rec)

    def sample(self, n):
        samples = []
        device = next(self.parameters()).device
        with torch.no_grad():
            # TODO: Sample from the model.
            # Generate n latent space samples and return their reconstructions.
            # Remember that for the model, this is like inference.
            # ====== YOUR CODE: ======
            samples = self.decode(torch.randn(n, self.z_dim, device=device))
            # ========================
        return samples

    def forward(self, x):
        z, mu, log_sigma2 = self.encode(x)
        return self.decode(z), mu, log_sigma2


def vae_loss(x, xr, z_mu, z_log_sigma2, x_sigma2):
    """
    Pointwise loss function of a VAE with latent space of dimension z_dim.
    :param x: Input image batch of shape (N,C,H,W).
    :param xr: Reconstructed (output) image batch.
    :param z_mu: Posterior mean (batch) of shape (N, z_dim).
    :param z_log_sigma2: Posterior log-variance (batch) of shape (N, z_dim).
    :param x_sigma2: Likelihood variance (scalar).
    :return:
        - The VAE loss
        - The data loss term
        - The KL divergence loss term
    all three are scalars, averaged over the batch dimension.
    """
    loss, data_loss, kldiv_loss = None, None, None
    # TODO: Implement the VAE pointwise loss calculation.
    # Remember that the covariance matrix of the posterior is diagonal.
    # ====== YOUR CODE: ======
    data_loss = F.mse_loss(xr, x, reduction='elementwise_mean') / x_sigma2
    kldiv_loss = -torch.mean(1 + z_log_sigma2 - z_mu.pow(2) - z_log_sigma2.exp())
    loss = data_loss + kldiv_loss
    # ========================

    return loss, data_loss, kldiv_loss
