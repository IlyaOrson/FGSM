# MLProb

The task involves developing a program that manipulates images by adding adversarial noise. This noise is designed to trick an image classification model into misclassifying the altered image as a specified target class, regardless of the original content.

You may select any pre-trained image classification model for this task. A model from the torchvision library is recommended, but not mandatory.
The core challenge is to effectively introduce noise into the image in such a way that the model misclassifies it as the desired target class, without making the noise perceptible to a casual human viewer.

![noisy_panda](noisy_panda.png)

Input:

The user will provide an image and specify a target class.

Output:

The program should output an image that has been altered with adversarial noise. The altered image should be classified by the model as the target class, irrespective of the original image's content. The altered image should not be obviously different to the original.

## Setup

We use [pixi](https://github.com/prefix-dev/pixi) to setup a reproducible environment with predefined tasks.
If you would like to use other project management tool, the list of dependencies and tasks are available in [pixi.toml](pixi.toml).

Clone this repo and install the dependencies of the project in a local environment.

```bash
git clone https://github.com/IlyaOrson/MLProb.git
cd MLProb
pixi install  # setup from pixi.toml file
```

Voila! An activated shell within this environment will have all dependencies working together.

<!-- ```bash
pixi shell  # activate shell
python run solution  # run main task
``` -->
