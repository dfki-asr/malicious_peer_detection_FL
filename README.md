### Detecting malicious updates with GANs in FL scenarios

* [FL scenarios](scenarios)

```bash
# To run a scenario
bash scenarios/scenario-id.sh
```
***Detection Strategy vs FedAvg: Additive noise attack (for 2 clients)***

![additive_noise](img/additive-noise-attack.png)

***Detection Strategy vs FedAvg: Sign flipping attack (for 2 clients)***

![sign_flipping](img/additive-noise-attack.png)

***Detection Strategy vs FedAvg: Same value attack (for 2 clients)***

![sign_flipping](img/same-value-attack.png)


* Visualizing training metrics
```bash
tensorboard --logdir ./fl_logs
```

![training_metrics](img/training_metrics.png)
![legend](img/legend.png)