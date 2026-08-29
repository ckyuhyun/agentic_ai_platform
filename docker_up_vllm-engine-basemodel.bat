docker compose -f docker-compose.yml --env-file ..\.env stop vllm-engine-basemodel
docker compose -f docker-compose.yml --env-file ..\.env rm -f vllm-engine-basemodel

docker compose -f docker-compose.yml --env-file ..\.env up -d --force-recreate vllm-engine-basemodel
