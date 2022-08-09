from ubuntu:20.04

RUN apt-get update && apt-get install -y curl gpg git
RUN curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | apt-key add
RUN echo "deb https://dl.yarnpkg.com/debian/ stable main" | tee /etc/apt/sources.list.d/yarn.list
RUN curl -fsSL https://deb.nodesource.com/setup_16.x | bash -


RUN apt-get update && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y curl python3 python3-pip nodejs yarn psmisc parallel

RUN npm i -g npm node-gyp@9.0.0 @mapbox/node-pre-gyp@1.0.8
RUN pip install web3 numpy scipy tabulate pytest networkx cachetools psycopg2-binary backoff pika bloom-filter2 leveldb

WORKDIR /opt/

RUN git clone --branch robmcl4/myFork --depth 1 https://github.com/robmcl4/ganache.git ganache-fork
RUN cd /opt/ganache-fork && npm install && INFURA_KEY=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa npm run build

WORKDIR /opt/goldphish

# build the shooter contract
COPY package.json .
COPY yarn.lock .
RUN yarn install

COPY hardhat.config.js .
COPY contracts contracts
RUN yarn hardhat compile

# copy in our ganache build
COPY . ./
