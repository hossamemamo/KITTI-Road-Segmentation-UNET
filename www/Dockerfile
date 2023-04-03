FROM python:3

WORKDIR /server

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY best_model_state.bin main.py ./

EXPOSE 5000

CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000" ]
