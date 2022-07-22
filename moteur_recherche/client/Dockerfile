FROM node:16.15.1
WORKDIR /client
ENV PATH="./node_modules/.bin:$PATH"
COPY ./package.json .
COPY ./package-lock.json .
RUN npm i
COPY . .
RUN npm run build
