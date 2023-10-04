from ubuntu:20.04

RUN apt-get update && apt-get install -y curl gpg git ca-certificates
RUN curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | apt-key add
RUN echo "deb https://dl.yarnpkg.com/debian/ stable main" | tee /etc/apt/sources.list.d/yarn.list

# set up nodesource Node v16
RUN mkdir -p /etc/apt/keyrings
RUN curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg
RUN echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_16.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list


RUN apt-get update && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y python3 python3-pip nodejs yarn psmisc parallel

RUN npm i -g npm@9.6.2 node-gyp@9.0.0 @mapbox/node-pre-gyp@1.0.8
WORKDIR /opt
COPY requirements.txt .
RUN pip install -r requirements.txt


RUN git clone --branch robmcl4/myFork  --depth 1 https://github.com/robmcl4/ganache.git ganache-fork
RUN cd /opt/ganache-fork && npm install && INFURA_KEY=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa npm run build

WORKDIR /opt/goldphish

# build the shooter contract
COPY package.json .
COPY yarn.lock .
RUN yarn install

COPY hardhat.config.js .
COPY contracts contracts
RUN yarn hardhat compile

COPY . ./
RUN python3 -m compileall -x node_modules .
