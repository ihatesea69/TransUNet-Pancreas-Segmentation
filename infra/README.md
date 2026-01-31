# Infrastructure Setup

## AWS CloudFormation Templates for TransUNet Training

This folder contains CloudFormation templates to deploy GPU instances for model training.

## Available Templates

### 1. `ec2-training-instance.yaml`
Deploys a single EC2 GPU instance with:
- NVIDIA GPU (g4dn.xlarge by default - 1x T4 GPU, 4 vCPU, 16GB RAM)
- Deep Learning AMI with PyTorch pre-installed
- 100GB EBS storage
- Security group for SSH access

## Quick Start

### Prerequisites
1. AWS CLI installed and configured
2. An existing EC2 Key Pair
3. AWS account with permissions to create EC2 instances

### Deploy Stack

```bash
# Deploy with default parameters
aws cloudformation create-stack \
  --stack-name transunet-training \
  --template-body file://ec2-training-instance.yaml \
  --parameters ParameterKey=KeyName,ParameterValue=YOUR_KEY_NAME \
  --capabilities CAPABILITY_IAM

# Deploy with custom instance type (for larger models)
aws cloudformation create-stack \
  --stack-name transunet-training \
  --template-body file://ec2-training-instance.yaml \
  --parameters \
    ParameterKey=KeyName,ParameterValue=YOUR_KEY_NAME \
    ParameterKey=InstanceType,ParameterValue=g4dn.2xlarge \
  --capabilities CAPABILITY_IAM
```

### Check Stack Status

```bash
aws cloudformation describe-stacks --stack-name transunet-training
```

### Get Instance Public IP

```bash
aws cloudformation describe-stacks \
  --stack-name transunet-training \
  --query 'Stacks[0].Outputs[?OutputKey==`PublicIP`].OutputValue' \
  --output text
```

### SSH to Instance

```bash
ssh -i your-key.pem ubuntu@<PUBLIC_IP>
```

### Delete Stack (when done)

```bash
aws cloudformation delete-stack --stack-name transunet-training
```

## Instance Types Comparison

| Instance Type | GPU | GPU Memory | vCPU | RAM | Cost/hour |
|--------------|-----|------------|------|-----|-----------|
| g4dn.xlarge | 1x T4 | 16 GB | 4 | 16 GB | ~$0.526 |
| g4dn.2xlarge | 1x T4 | 16 GB | 8 | 32 GB | ~$0.752 |
| g4dn.4xlarge | 1x T4 | 16 GB | 16 | 64 GB | ~$1.204 |
| g5.xlarge | 1x A10G | 24 GB | 4 | 16 GB | ~$1.006 |
| g5.2xlarge | 1x A10G | 24 GB | 8 | 32 GB | ~$1.212 |
| p3.2xlarge | 1x V100 | 16 GB | 8 | 61 GB | ~$3.06 |

## Training Setup on Instance

After SSH into the instance:

```bash
# Clone repository
git clone https://github.com/ihatesea69/TransUNet-Pancreas-Segmentation.git
cd TransUNet-Pancreas-Segmentation

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install monai nibabel tqdm matplotlib scipy

# Download dataset
python scripts/download_dataset.py

# Run training (use screen/tmux for long-running tasks)
screen -S training
cd notebooks
jupyter nbconvert --to script 03_Training_Pipeline.ipynb
python 03_Training_Pipeline.py

# Or run Jupyter Notebook
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
```

## Cost Optimization Tips

1. **Use Spot Instances**: Add `SpotPrice` parameter for up to 70% savings
2. **Stop when not training**: Use `aws ec2 stop-instances` to pause billing
3. **Use smaller instance for debugging**: Start with g4dn.xlarge, upgrade if needed
4. **Set up auto-shutdown**: Configure CloudWatch alarm to stop idle instances
