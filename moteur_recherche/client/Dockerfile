FROM node:16.15.1
WORKDIR /client
COPY ./package.json .
RUN npm install
COPY . .
CMD ["npm","start"]