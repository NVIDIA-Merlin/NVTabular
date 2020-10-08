docker build -t rapidsdl_gds -f ./examples/hugectr/docker/Dockerfile.gds .
docker run --runtime=nvidia -p 38888:8888 -p 38787:8787 -p 38786:8786 -p 33000:3000 --ipc=host --name gds_test rapidsdl_gds /bin/bash -c "cd / && chmod +777 int_setup.sh && ./int_setup.sh"
docker commit $(docker ps -aqf "name=^gds_test$") gds_rel
