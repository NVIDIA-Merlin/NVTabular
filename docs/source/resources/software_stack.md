We offer the following software stacks:

- <a href="#merlin_inference">Merlin Inference</a>: Allows you to deploy NVTabular workflows and HugeCTR or TensorFlow models to the Triton Inference server for production.
- <a href="#merlin_training">Merlin Training</a>: Allows you to do preprocessing and feature engineering with NVTabular so that you can train a deep learning recommendation model with HugeCTR.
- <a href="#merlin_tensorflow_training">Merlin TensorFlow Training</a>: Allows you to do preprocessing and feature engineering with NVTabular so that you can train a deep learning recommendation model with TensorFlow.
- <a href="#merlin_pytorch_training">Merlin PyTorch Training</a>: Allows you to do preprocessing and feature engineering with NVTabular so that you can train a deep learning recommendation model with PyTorch.

The following tables provide the software and model versions that NVTabular version 0.6 supports.

<div align="center"><a name="merlin_inference">Table 1: Software stack matrix for the Merlin Inference (merlin-inference) image</a></div>
<br>

<table style="align:center">
  <tr>
    <td colspan="2"><b>DGX</b></td>
  </tr>
  <tr>
    <td>DGX System</td>
    <td>
      <ul>
        <li>DGX-1</li>
        <li>DGX-2</li>
        <li>DGX A100</li>
        <li>DGX Station</li>
      </ul>
    </td>
  <tr>
    <td>Operating System</td>
    <td>Ubuntu version 20.04</td>
  </tr>
  <tr>
    <td colspan="2"><b>NVIDIA Certified Systems</b></td>
  </tr>
  <tr>
    <td>NVIDIA Driver</td>
    <td>
      <p>The 21.06 release is based on NVIDIA CUDA version 11.3.1, which requires NVIDIA Driver version 465.19.01 or later. However, if you are running on Data Center GPUs (formerly Tesla) such as T4, you can use any of the following NVIDIA Driver versions:
        <ul>
          <li>418.40 (or later R418)</li>
          <li>440.33 (or later R440)</li>
          <li>450.51 (or later R450)</li>
          <li>460.27 (or later R460)</li>
        </ul>
      </p>
      <p><b>NOTE</b>: The CUDA Driver Compatibility Package doesn’t support all drivers.</p>
    </td>
  </tr>
  <tr>
    <td>GPU Model</td>
    <td>
      <ul>
        <li><a href="https://www.nvidia.com/en-us/data-center/ampere-architecture/">NVIDIA Ampere GPU Architecture</a>   
        </li>
        <li><a href="https://www.nvidia.com/en-us/geforce/turing/">Turing</a></li>
        <li><a href="https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/">Volta</a></li>
        <li><a href="https://www.nvidia.com/en-us/data-center/pascal-gpu-architecture/">Pascal</a></li>
      </ul>
    </td>
  </tr>
  <tr>
    <td colspan="2"><b>Base Container Image</b></td>
  </tr>
  <tr>
    <td>Container OS</td>
    <td>Ubuntu version 20.04</td>
  </tr>
  <tr>
    <td>Base Container</td>
    <td>Triton version 21.06</td>
  </tr>
  <tr>
    <td>CUDA</td>
    <td>11.3</td>
  </tr>
  <tr>
    <td>RMM</td>
    <td>21.06</td>
  </tr>
  <tr>
    <td>cuDF</td>
    <td>21.06</td>
  </tr>
  <tr>
    <td>cuDNN</td>
    <td>N/A</td>
  </tr>
  <tr>
    <td>HugeCTR</td>
    <td>3.1</td>
  </tr>
  <tr>
    <td>NVTabular</td>
    <td>0.6</td>
  </tr>
</table>
<br>

<div align="center"><a name="merlin_training">Table 2: Software stack matrix for the Merlin Training (merlin-training) image</a></div>
<br>

<table style="align:center">
  <tr>
    <td colspan="2"><b>DGX</b></td>
  </tr>
  <tr>
    <td>DGX System</td>
    <td>
      <ul>
        <li>DGX-1</li>
        <li>DGX-2</li>
        <li>DGX A100</li>
        <li>DGX Station</li>
      </ul>
    </td>
  <tr>
    <td>Operating System</td>
    <td>Ubuntu version 20.04</td>
  </tr>
  <tr>
    <td colspan="2"><b>NVIDIA Certified Systems</b></td>
  </tr>
  <tr>
    <td>NVIDIA Driver</td>
    <td>
      <p>The 21.06 release is based on NVIDIA CUDA version 11.3.1, which requires NVIDIA Driver version 465.19.01 or later. However, if you are running on Data Center GPUs (formerly Tesla) such as T4, you can use any of the following NVIDIA Driver versions:
        <ul>
          <li>418.40 (or later R418)</li>
          <li>440.33 (or later R440)</li>
          <li>450.51 (or later R450)</li>
          <li>460.27 (or later R460)</li>
        </ul>
      </p>
      <p><b>NOTE</b>: The CUDA Driver Compatibility Package doesn’t support all drivers.</p>
    </td>
  </tr>
  <tr>
    <td>GPU Model</td>
    <td>
      <ul>
        <li><a href="https://www.nvidia.com/en-us/data-center/ampere-architecture/">NVIDIA Ampere GPU Architecture</a>   
        </li>
        <li><a href="https://www.nvidia.com/en-us/geforce/turing/">Turing</a></li>
        <li><a href="https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/">Volta</a></li>
        <li><a href="https://www.nvidia.com/en-us/data-center/pascal-gpu-architecture/">Pascal</a></li>
      </ul>
    </td>
  </tr>
  <tr>
    <td colspan="2"><b>Base Container Image</b></td>
  </tr>
  <tr>
    <td>Container OS</td>
    <td>Ubuntu version 20.04</td>
  </tr>
  <tr>
    <td>Base Container</td>
    <td>N/A</td>
  </tr>
  <tr>
    <td>CUDA</td>
    <td>11.3</td>
  </tr>
  <tr>
    <td>RMM</td>
    <td>21.06</td>
  </tr>
  <tr>
    <td>cuDF</td>
    <td>21.06</td>
  </tr>
  <tr>
    <td>cuDNN</td>
    <td>8.2.2</td>
  </tr>
  <tr>
    <td>HugeCTR</td>
    <td>N/A</td>
  </tr>
  <tr>
    <td>NVTabular</td>
    <td>0.6</td>
  </tr>
</table>
<br>

<div align="center"><a name="merlin_tensorflow_training">Table 3: Software stack matrix for the Merlin TensorFlow Training (merlin-tensorflow-training) image</a></div>
<br>

<table style="align:center">
  <tr>
    <td colspan="2"><b>DGX</b></td>
  </tr>
  <tr>
    <td>DGX System</td>
    <td>
      <ul>
        <li>DGX-1</li>
        <li>DGX-2</li>
        <li>DGX A100</li>
        <li>DGX Station</li>
      </ul>
    </td>
  <tr>
    <td>Operating System</td>
    <td>Ubuntu version 20.04</td>
  </tr>
  <tr>
    <td colspan="2"><b>NVIDIA Certified Systems</b></td>
  </tr>
  <tr>
    <td>NVIDIA Driver</td>
    <td>
      <p>The 21.06 release is based on NVIDIA CUDA version 11.3.1, which requires NVIDIA Driver version 465.19.01 or later. However, if you are running on Data Center GPUs (formerly Tesla) such as T4, you can use any of the following NVIDIA Driver versions:
        <ul>
          <li>418.40 (or later R418)</li>
          <li>440.33 (or later R440)</li>
          <li>450.51 (or later R450)</li>
          <li>460.27 (or later R460)</li>
        </ul>
      </p>
      <p><b>NOTE</b>: The CUDA Driver Compatibility Package doesn’t support all drivers.</p>
    </td>
  </tr>
  <tr>
    <td>GPU Model</td>
    <td>
      <ul>
        <li><a href="https://www.nvidia.com/en-us/data-center/ampere-architecture/">NVIDIA Ampere GPU Architecture</a>   
        </li>
        <li><a href="https://www.nvidia.com/en-us/geforce/turing/">Turing</a></li>
        <li><a href="https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/">Volta</a></li>
        <li><a href="https://www.nvidia.com/en-us/data-center/pascal-gpu-architecture/">Pascal</a></li>
      </ul>
    </td>
  </tr>
  <tr>
    <td colspan="2"><b>Base Container Image</b></td>
  </tr>
  <tr>
    <td>Container OS</td>
    <td>Ubuntu version 20.04</td>
  </tr>
  <tr>
    <td>Base Container</td>
    <td><a href="https://nvcr.io/nvidia/tensorflow:21.06-tf2-py3">nvcr.io/nvidia/tensorflow:21.06-tf2-py3</a></td>
  </tr>
  <tr>
    <td>CUDA</td>
    <td>11.3</td>
  </tr>
  <tr>
    <td>RMM</td>
    <td>21.06</td>
  </tr>
  <tr>
    <td>cuDF</td>
    <td>21.06</td>
  </tr>
  <tr>
    <td>cuDNN</td>
    <td>N/A</td>
  </tr>
  <tr>
    <td>HugeCTR</td>
    <td>3.1</td>
  </tr>
  <tr>
    <td>NVTabular</td>
    <td>0.6</td>
  </tr>
</table>
<br>

<div align="center"><a name="merlin_pytorch_training">Table 4: Software stack matrix for the Merlin PyTorch Training (merlin-pytorch-training) image</a></div>
<br>

<table style="align:center">
  <tr>
    <td colspan="2"><b>DGX</b></td>
  </tr>
  <tr>
    <td>DGX System</td>
    <td>
      <ul>
        <li>DGX-1</li>
        <li>DGX-2</li>
        <li>DGX A100</li>
        <li>DGX Station</li>
      </ul>
    </td>
  <tr>
    <td>Operating System</td>
    <td>Ubuntu version 20.04</td>
  </tr>
  <tr>
    <td colspan="2"><b>NVIDIA Certified Systems</b></td>
  </tr>
  <tr>
    <td>NVIDIA Driver</td>
    <td>
      <p>The 21.06 release is based on NVIDIA CUDA version 11.3.1, which requires NVIDIA Driver version 465.19.01 or later. However, if you are running on Data Center GPUs (formerly Tesla) such as T4, you can use any of the following NVIDIA Driver versions:
        <ul>
          <li>418.40 (or later R418)</li>
          <li>440.33 (or later R440)</li>
          <li>450.51 (or later R450)</li>
          <li>460.27 (or later R460)</li>
        </ul>
      </p>
      <p><b>NOTE</b>: The CUDA Driver Compatibility Package doesn’t support all drivers.</p>
    </td>
  </tr>
  <tr>
    <td>GPU Model</td>
    <td>
      <ul>
        <li><a href="https://www.nvidia.com/en-us/data-center/ampere-architecture/">NVIDIA Ampere GPU Architecture</a>   
        </li>
        <li><a href="https://www.nvidia.com/en-us/geforce/turing/">Turing</a></li>
        <li><a href="https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/">Volta</a></li>
        <li><a href="https://www.nvidia.com/en-us/data-center/pascal-gpu-architecture/">Pascal</a></li>
      </ul>
    </td>
  </tr>
  <tr>
    <td colspan="2"><b>Base Container Image</b></td>
  </tr>
  <tr>
    <td>Container OS</td>
    <td>Ubuntu version 20.04</td>
  </tr>
  <tr>
    <td>Base Container</td>
    <td><a href="https://nvcr.io/nvidia/pytorch:21.06-py3">nvcr.io/nvidia/pytorch:21.06-py3</a></td>
  </tr>
  <tr>
    <td>CUDA</td>
    <td>11.3</td>
  </tr>
  <tr>
    <td>RMM</td>
    <td>21.06</td>
  </tr>
  <tr>
    <td>cuDF</td>
    <td>21.06</td>
  </tr>
  <tr>
    <td>cuDNN</td>
    <td>N/A</td>
  </tr>
  <tr>
    <td>HugeCTR</td>
    <td>N/A</td>
  </tr>
  <tr>
    <td>NVTabular</td>
    <td>0.6</td>
  </tr>
</table>
