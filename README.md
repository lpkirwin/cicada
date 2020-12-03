# Cicada

This repo contains code used for my submissions to the Kaggle football competition in November 2020.

It started out as a rules-based agent, but then I added some ML components by scoring move/pass options based on a modelled probability of success.

## Setup

https://github.com/google-research/football

Mac

- `brew install git python3 cmake sdl2 sdl2_image sdl2_ttf sdl2_gfx boost boost-python3`
- `pip3 install gfootball`
- `python3 -m gfootball.play_game --action_set=full`

Linux

- Used kaggle docker image
  