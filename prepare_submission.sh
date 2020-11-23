#!/bin/bash
now=$(date +%Y%m%d%H%M)
echo "writing to submissions/cicada_$now.tar.gz"
rm -r submissions/cicada
mkdir submissions/cicada
mkdir submissions/cicada/cicada
mkdir submissions/cicada/cicada/utils
mkdir submissions/cicada/models
cp main.py submissions/cicada/
cp cicada/agent.py submissions/cicada/cicada/
cp cicada/utils/*.py submissions/cicada/cicada/utils/
cp models/*.txt submissions/cicada/models/
tar -czvf submissions/cicada_$now.tar.gz -C submissions/cicada/ .
