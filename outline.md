## initial plan 

- originally intended to implement PIVO or HybVIO 
- PIVO: Probabilistic Inertial-Visual Odometry for Occlusion-Robust Navigation
- HybVIO: Pushing the Limits of Real-time Visual-inertial Odometry
- but derivation is not clear, not mentioned in paper 
- eg page 11, eq. A3, Quaternion update by angular velocity part is confusing 
- several questions regarding derivation in https://github.com/SpectacularAI/HybVIO/issues
![img](/assets/a3.png)

## msckf 

- build doc 
- load dataset 
    - use script to generate video 
    - load imu and frame 
- initialize orientation 
- imu prediction 
    - quaternion update, why -0.5 
- visual update 
    - fast detector 
- next steps 
    - add graph optimization, eg https://github.com/code-cp/vio-course-rust/tree/main/ch03/rust 
    - add dynamic init 
    - add FEJ 

