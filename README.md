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

- [ ] Try making pos_score for pass based on timestep + 5
  - May have to start training from scratch :(
- [x] Manually increasing value of dopp and kopp in position score
  - Tried, didn't really work :(
- [x] Increasing negative reward for losing possession
- [ ] Try game weight of 1.0
- [ ] Different levels of passing bonus
- [ ] Improve pass_score
- [x] Last minute breakaway pass
- [x] Shorten timesteps for passes
- [ ] Is negative effect of opp_poss in pos_score too strong?

## Submission notes

cicada_202011251854 = unknown
cicada_202011252100 = unknown
cicada_202011252325 = second submission to kaggle, fixed shooting, also implemented multi-step kicks
cicada_202011260004 = v3 position score
cicada_202011261101 = shortened kick timestep -> 3
cicada_202011261226 = implemented breakaway decision point
cicada_202011261439 = new v3 position score with negative signal reduced 10x
cicada_202011270003 = using old handle model
cicada_202011272336 = rolled back to old handle dataset, also implemented length-aware passing
cicada_202011272338 = same but with hardcoded shot value
cicada_202011280851 = same but increased propensity to shoot
cicada_202011280853 = same but v2 position score
cicada_202011291711 = after overnight training and some extra pass features

## Tournaments

cicada_202011252325    1.655556    0.544444
cicada_202011260004    1.561111    0.538889
cicada_202011252100    1.544444    0.461111
cicada_202011251854    1.344444    0.455556

cicada_202011260004    1.68    0.635
cicada_202011252100    1.55    0.550
cicada_202011261226    1.26    0.525
cicada_202011261101    1.49    0.520
cicada_202011252325    1.56    0.495
cicada_202011251854    1.28    0.480
cicada_202011261439    1.01    0.410
cicada_202011270003    1.04    0.385

cicada_202011272338    1.661538    0.615385
cicada_202011272336    1.484615    0.538462
cicada_202011252100    1.507692    0.492308
cicada_202011252325    1.500000    0.492308
cicada_202011251854    1.423077    0.465385
cicada_202011260004    1.353846    0.480769
cicada_202011261101    1.300000    0.476923
cicada_202011261226    1.230769    0.438462

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