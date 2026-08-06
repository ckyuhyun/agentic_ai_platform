docker compose -f docker-compose.yml stop vllm-engine-finetunedmodel
docker compose -f docker-compose.yml stop vllm-engine-basemodel
docker compose -f docker-compose.yml rm -f vllm-engine-finetunedmodel vllm-engine-basemodel

docker compose -f docker-compose.yml up -d --force-recreate vllm-engine-basemodel
