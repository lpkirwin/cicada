# Cicada

## Setup

https://github.com/google-research/football

Mac

  - `brew install git python3 cmake sdl2 sdl2_image sdl2_ttf sdl2_gfx boost boost-python3`
  - `pip3 install gfootball`
  - `python3 -m gfootball.play_game --action_set=full`
  - `python3 -m gfootball.play_game --action_set=full --level=academy_counterattack_hard`

Linux

## Todo

[ ] Try making pos_score for pass based on timestep + 5
  - May have to start training from scratch :(
[x] Manually increasing value of dopp and kopp in position score
  - Tried, didn't really work :(
[x] Increasing negative reward for losing possession

## Notes

short
- mean 38.6, p95 69.9
- mean 41.5, p95 75.8
- mean 41.6, p95 76.1

long
- mean 32.3, p95 63.0
- mean 34.5, p95 66.3

Training:

- Starting out with fixed success prbs
- Boosting pass success prbs by 10%