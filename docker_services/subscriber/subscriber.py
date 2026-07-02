import redis
import os
import json
from dotenv import load_dotenv


load_dotenv()
redis_host = os.getenv('Redis_Host')
redis_port = os.getenv('Redis_Port')
redis_password = os.getenv('Redis_Password')
redis_default_key_name = os.getenv('Redis_Default_Key')



def redis_conn():
    try:
        redis_client = redis.Redis(host=redis_host,
                            port=redis_port,
                            password=redis_password,
                            db=0,
                            decode_responses=True)
    except Exception as e:
        print(f"[Error] Failed to connect to Redis: {e}")
        raise e
    
    return redis_client



def subscriber():
    redis_client = redis_conn()

    while True:
        try:
            result = redis_client.blpop(keys=redis_default_key_name, timeout=3)
        except Exception as e:
            continue

        
        if result:
            _, raw_event = result

            event_dict = json.loads(raw_event)
            event_id = event_dict.get('event_id')
            event_data = event_dict.get('data', {})

            print(f"[*] Processing Event: {event_id}")

            content = event_data.get("content")
            project_id = event_data.get("project_id")
            meta_data = event_data.get("metadata", {})

            # De-duplication check
            dedup_key = f"dedup:{hash(content)}"
            if redis_client.get(dedup_key):
                print(f"[Warning] Duplicate event skipped for ID {event_id}")

                continue

            # set a 30 secs TTL to forget this payload signature
            redis_client.set(dedup_key, "processed", ex=30)

            

        

        

    


if __name__ == "__main__":
    subscriber()