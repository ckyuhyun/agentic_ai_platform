import uvicorn
import os

from fastapi import FastAPI, status, Request
from dotenv import load_dotenv



app = FastAPI()



@app.post('/update',
          status_code=status.HTTP_200_OK)
async def update(request :Request):


    return status.HTTP_200_OK




if __name__ == "__main__":
    load_dotenv()
    host = os.getenv("POSTGRES_HOST")
    port = int(os.getenv("POSTGRES_POST"))
    
    uvicorn.run(app, host="0.0.0.0", port=8080)