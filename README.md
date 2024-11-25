# AutoBS - Inference

Inference code of the AutoBS framework from our paper "AutoBS: Autonomous Base Station Deployment Framework with Reinforcement Learning and Digital Twin Network".
<div>
<img src="figures/animation_autobs.gif" alt="animation_autobs" width="500" />
</div>

## Highlights
- We introduce a novel DRL-based framework for single/multi-BS deployment that incorporates [PMNet](https://arxiv.org/abs/2312.03950) for real-time, site-specific channel predictions.
- AutoBS reduces inference time from hours to milliseconds compared to exhaustive methods, particularly in multi-BS deployments, making it practical for large-scale, real-time optimization.
- The repository includes the checkpoints of our single-BS, multi-BS agent and the PMNet. *SionnaRT* is used for visualizing the deployment result.


## Citation

```

```


## Available checkpoints

| Model           | Download Link |
|-----------------|---------------|
| Single-BS Agent | [Download]()  |
| Multi-BS Agent  | [Download]()  |
| PMNet           | [Download]()  |

## Inference

To evaluate the performance of our AutoBS agent, refer to the following commands to deploy either a single base station or two base stations on a test map. Note that this script would also execute the heuristic and exhaustive methods for comparison.

```bash
python inference.py \
    --version [single/multi] \
    --crop_id [test-map-id] \ # enter a number between 0 and 15
    --reward_type [coverage/capacity] \ #  for baseline methods
# e.g.,
# python inference.py \
#    --version single \
#    --crop_id 0 \
#    --reward_type coverage \
```

After running the inference script, the output coverage map will be saved in the `visulaize/sionna_output/` directory.


