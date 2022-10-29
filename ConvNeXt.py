import torch
import torchvision

class ConvNeXtBlock(torch.nn.Module):
  def __init__(self, in_channels, out_channels, expansion_ratio, dropout=0.1):
    super().__init__()
    hidden_dim = in_channels * expansion_ratio

    # Depthwise Convolution
    self.spatial_mixing = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels),
        torch.nn.BatchNorm2d(in_channels)
    )

    # Pointwise Convolution, Upsampling Channels
    self.feature_mixing = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0),
        torch.nn.GELU(),
    )

    # Pointwise convolution, Downsampling Channels
    self.bottleneck = torch.nn.Sequential(
        torch.nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0)
    )

    # Drop Path / Stochastic Depth
    self.stochastic_depth = torchvision.ops.StochasticDepth(p=dropout, mode="batch") 

  def forward(self, x):
    out = self.spatial_mixing(x)
    out = self.feature_mixing(out)
    out = self.bottleneck(out)

    # Residual Connection within Block
    # out_channels of each ConvNeXtBlock is the same as in_channels, so we have a residual connection added without any upsampling
    return x + self.stochastic_depth(out)

class ConvNeXt(torch.nn.Module):

  def __init__(self, num_classes, in_channels, stage_config):
    super().__init__()
    self.num_classes = num_classes
    self.in_channels = in_channels
    
    # Define Stage Configurations
    self.stage_config = stage_config

    self.layers = []

    # Define Stem (Kernel Size 4, Stride 4)
    stem = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, self.stage_config[0][1], kernel_size=4, stride=4),
        torch.nn.BatchNorm2d(96)
    )
    
    self.layers.append(stem)
  
    
    # Define Each Stage
    for i in range(0, len(self.stage_config)):
  
      # Create stage
      expansion_ratio, in_channels, num_blocks, dropout = self.stage_config[i]

      # Append Blocks Depth Number of Times 
      for j in range(0, num_blocks):
        self.layers.append(
            ConvNeXtBlock(in_channels=in_channels, out_channels=in_channels, 
                          expansion_ratio=expansion_ratio, dropout=dropout)
            )
        
      # Append downsampling layers at the end of each stage
      if i < len(self.stage_config)-1:
        # Append downsampling layer
        next_stage_in_channels = self.stage_config[i+1][1]
        dsl_layer = torch.nn.Sequential(
          torch.nn.Conv2d(in_channels, next_stage_in_channels, kernel_size=2, stride=2),
          torch.nn.BatchNorm2d(next_stage_in_channels),
        )
        self.layers.append(dsl_layer)
    
    self.layers = torch.nn.Sequential(*self.layers)

    # Embeddings
    self.embeddings = torch.nn.Sequential(
        torch.nn.BatchNorm2d(self.stage_config[-1][1]),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        # Optional - Activation (GELU?)
    )

    # Classification Layer
    self.head = torch.nn.Linear(self.stage_config[-1][1], self.num_classes)

  def forward(self, x):
    out = self.layers(x)
    out = self.embeddings(out)
    out = self.head(out)
    return out 