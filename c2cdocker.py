import os
import argparse
import shutil


def make_tar_copy(tar_path, name):
    name = f"Dockerfile.{name}"
    shutil.copyfile(tar_path, name)
    return name

def chg_ln(cur_line, new_line, tar_path):
    cmd = f"sed -i 's/{cur_line}[^\\\"]*/{new_line}/' {tar_path}"
    os.system(cmd)

def create_cont(cuda_toolkit, v_python, v_torch=None, v_tf=None, tar_path='Dockerfile.nvtab', upload_locs=['gitlab-master.nvidia.com:5005/rapidsdl/docker/'], upload_creds=[]):
    name = f"{cuda_toolkit.replace('.', '-')}_{v_python.replace('.', '-')}"
    if v_torch:
        name = name + f"_torch-{v_torch.replace('.', '-')}"
    if v_tf:
        name = name + f"_tf-{v_tf.replace('.', '-')}"
    new_dock = make_tar_copy(tar_path, name)
    tar_path = new_dock
    chg_ln('ARG CUDA_VERSION', f'ARG CUDA_VERSION={cuda_toolkit}', tar_path)
    chg_ln('ARG PYTHON_VERSION', f'ARG PYTHON_VERSION={v_python}', tar_path)
    d_file = open(tar_path, 'a')
    if v_torch:
        d_file.write('RUN ' + add_torch(v_torch, cuda_toolkit) + '\n')
    if v_tf:
        d_file.write('RUN ' + add_tf(v_tf, cuda_toolkit) + '\n')
    d_file.close()
    execute_build(tar_path, name)
    upload_container(name, upload_locs, upload_creds)


def add_torch(v_torch, cuda_toolkit):
    cmd = f'conda install pytorch={v_torch} torchvision cudatoolkit={cuda_toolkit} -c pytorch'
    return cmd

def add_tf(v_tf, cuda_toolkit):
    cmd = f'pip install --upgrade grpcio==1.24.3 tensorflow-gpu=={v_tf} tfdlpack-gpu'
    if float(cuda_toolkit) > 10.1:
        cmd = cmd + '; ln -s /usr/local/cuda/lib64/libcudart.so.10.2 /usr/local/cuda/lib64/libcudart.so.10.1'
    return cmd


def execute_build(dockerfile, tag):
    cmd = f"docker build -t {tag} -f {dockerfile} ."
    os.system(cmd)
    print('New container built: ' + tag)


def upload_container(name, upload_locs, upload_creds):
    for idx, loc in enumerate(upload_locs):
        user = upload_creds[idx][0]
        passw = upload_creds[idx][1]
        base_loc = loc.split('/')[0]
        cmd = f"docker login -u {user} -p {passw} {base_loc}"
        os.system(cmd)
        cmd = f"docker tag {name} {loc}{name}:latest"
        os.system(cmd)
        cmd = f"docker push {loc}{name}:latest"
        os.system(cmd)



def main(args):
    create_cont(args.cuda_toolkit, args.python, v_torch=args.torch, v_tf=args.tf, upload_locs=args.upload_locs, upload_creds=args.upload_creds)


def parse_args():
    parser = argparse.ArgumentParser(description="Customize dockerfile to create desired container")
    parser.add_argument("--cuda_toolkit", type=str, help="the cuda toolkit version of the container")
    parser.add_argument("--python", type=str, help="desired version of python on container")
    parser.add_argument("--torch", type=str, help="version of pytorch to install")
    parser.add_argument("--tf", type=str, help="version of tensorflow-gpu to install")
    parser.add_argument("--tar_path", type=str, help="target dockerfile to edit", required=False)
    parser.add_argument("--upload_locs", type=str, help="list of locations to upload container too", required=False)
    parser.add_argument("--upload_creds", type=str, help='list of tuples of credentials, match order of upload_locs, format user:pass,user:pass,...')
    args = parser.parse_args()
    args.upload_locs = args.upload_locs.split(',')
    creds_tups = []
    for tup in args.upload_creds.split(','):
        to_add = tup.split(':')
        to_add = (to_add[0], to_add[1])
        creds_tups.append(to_add)
    args.upload_creds = creds_tups
    return args


if __name__=='__main__':
    main(parse_args())
