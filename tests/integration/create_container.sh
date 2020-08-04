docker run --runtime=nvidia -p 38888:8888 -p 38787:8787 -p 38786:8786 -p 33000:3000 --ipc=host --name hugectr_test rapidsdl_hugectr /bin/bash -c "source activate base && cd / && chmod +777 hugectr_setup.sh && ./hugectr_setup.sh"
docker commit $(docker ps -aqf "name=^hugectr_test$") hugectr_rel
