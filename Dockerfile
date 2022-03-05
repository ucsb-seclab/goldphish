from ubuntu:20.04

RUN apt-get update && apt-get install -y curl gpg
RUN curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | apt-key add
RUN echo "deb https://dl.yarnpkg.com/debian/ stable main" | tee /etc/apt/sources.list.d/yarn.list
RUN curl -fsSL https://deb.nodesource.com/setup_17.x | bash -


RUN apt-get update && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y python3 python3-pip nodejs yarn psmisc

RUN pip install web3 numpy scipy tabulate pytest networkx

WORKDIR /opt/goldphish

# build the shooter contract
COPY package.json .
COPY yarn.lock .
RUN yarn install

COPY hardhat.config.js .
COPY contracts contracts
RUN yarn hardhat compile

# COPY vend vend

# # build ganache
# RUN cd vend/ganache && npm install
# RUN cd vend/ganache && npm ci

COPY . ./
