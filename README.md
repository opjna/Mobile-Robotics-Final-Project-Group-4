# A Flexible Framework for Error Modeling and Uncertainty Propagation using a Generalized Skew-𝑡 Distribution

Final project for Group 4 of Winter 2022 ROB530: Mobile Robotics. This captures the code used to demonstrate the extension of the Generalized Unscented Transform ([GenUT](https://github.com/Schiff-Lab/Generalized-Unscented-Transform)) to skew-t error distributions. We show that a skew-t estimate can be found using a Maximum Likelihood Estimator, and that this estimated can be used with the GenUT approach to calculate appropriate sigma points for an unscented transform. Finally, we demonstrate the applicability of the proposed approach on simple simulated datasets, real IMU data, and estimated position data for Visual Odometry purposes.

Some of the code in this repo references the KITTI dataset, which can be found [here](http://www.cvlibs.net/datasets/kitti/eval_odometry.php).

All of the code included here is python. Dependencies include: numpy, Open CV2, matplotlib, scipy, pandas, PIL, numba.

## To use our software:

### Create virtual environment
```
python3 -m venv mobrob-venv
source mobrob-venv/bin/activate
```

### Install requirements
```
pip install -r requirements.txt
```

### Run code
```
python -i src/distribution_tools/tmain.py
```




