Клонировать репозиторий:
git clone https://github.com/aclbk54/U-2-Net.git


Скачать веса по ссылке https://drive.google.com/file/d/1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ/view?usp=sharing
и положить в папку saved_models файл .pth

docker build -t u2_net_test .


docker run --rm -v "$(pwd)/data:/app/data" u2_net_test
