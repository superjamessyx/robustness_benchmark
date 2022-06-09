Code for [Benchmarking the Robustness of Deep Neural Networks to Common  Corruptions in Digital Pathology]. 
This work can be used to evaluate how deep neural networks perform on corrupted pathology images.
An example is show in `test.py`.

Applying our benchmark construction method to your own dataset only 
needs to define DistortDataset like the following usage.  

**Usage**:

```
from utils import DistortImageFolder

distorted_dataset = DistortImageFolder(root='validation/set/root/of/Your/dataset/', method=distortion_name, severity=severity,
            transform=test_transform)
```

Note that the parameter setting
of corruption cannot be appropriate for all tasks. Thus you can adjust the
parameter to ensure that the corruption is close to reality and does not 
destroy all pathological information.

`plot_corrupted_sample.py` can be used to check whether the parameter
is  reasonable  by visualizing one example. `LocalTCT_vis_sample.png` and `Patchcamelyon_vis_sample.png` are two examples
in our dataset.