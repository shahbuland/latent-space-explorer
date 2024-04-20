# Latent Space Exploration

You can find cool things moving between encoded prompts in a conditional diffusion models latent space. This repository is for doing that. 

# Usage  

`python -m main`  
  
Run the above command to explore. Feel free to mess around with the config to change how sampling is done, font sizes, etc.  
Controls:  
Q/E: zoom out/in  
WASD: move up/left/down/right  
Click anywhere to generate a new image  
Click an existing point/node to select it  
T: modify prompt in selected node  
O: delete selected node  
P: create a new node  
Right click an existing point/node to move it around  
M: Switch between sampling modes
  
# Notes  

The zero vector for the encoding space corresponds to a kind of "garden". This corresponds to the negative prompt used in CFG. You can see this at the center of circle mode or when going very far from some points in distance mode.  
  
In circle mode, having multiple prompts on the opposite side from your generation results in heavy negative coefficients which often knocks generation into garbage area of latent space.  
To stay inside space of plausible generations, try to balance all sides of the circle.   
  
In distance mode, your generation will go towards the zero vector if it is not close to any points, because it falls of to 0 with inverse of square distance to each point.
  
# Sampling Modes  
  
Distance sampling assigns weights to each encoding based on its square distance from your control point  
  
Circle sampling assigns weights to each encoding based on cosine similarity with your control point. Norm is factored in so this is basically just a dot product.
