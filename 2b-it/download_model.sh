#!/bin/bash
# Export your Kaggle username and API key
# export KAGGLE_USERNAME=<YOUR USERNAME>
# export KAGGLE_KEY=<YOUR KAGGLE KEY>

curl -L -u $KAGGLE_USERNAME:$KAGGLE_KEY\
  -o ~/2b-it/model.tar.gz\
  https://www.kaggle.com/api/v1/models/google/recurrentgemma/flax/2b-it/1/download



  export KAGGLE_USERNAME=mehulbafnaa

  export KAGGLE_KEY=450080c1c9fbe22c908798f4f05bf9c4
