#!/bin/bash
now=$(date +%Y%m%d%H%M)
echo "writing to submissions/cicada_$now.tar.gz"
rm -r submissions/cicada
mkdir submissions/cicada
mkdir submissions/cicada/cicada_$now
mkdir submissions/cicada/cicada_$now/utils
mkdir submissions/cicada/models
cp main.py submissions/cicada/
cp cicada/agent.py submissions/cicada/cicada_$now/
cp cicada/utils/*.py submissions/cicada/cicada_$now/utils/
cp models/*.txt submissions/cicada/models/
sed -i.tmp "s/cicada/cicada_$now/g" submissions/cicada/*.py
sed -i.tmp "s/cicada/cicada_$now/g" submissions/cicada/cicada_$now/*.py
sed -i.tmp "s/cicada/cicada_$now/g" submissions/cicada/cicada_$now/utils/*.py
rm submissions/cicada/*.py.tmp
rm submissions/cicada/cicada_$now/*.py.tmp
rm submissions/cicada/cicada_$now/utils/*.py.tmp
tar -czvf submissions/cicada_$now.tar.gz -C submissions/cicada/ .
