# pokemon-type-prediction

![Pokemon Logo](https://cdn.bulbagarden.net/upload/d/d2/Pok%C3%A9mon_logo_English.png)



### Small project utilizing three supervised learning models to predict Pokemon types:

1. Naives Bayes
2. Support Vector Machines
3. Random Forest

### In addition, the following techniques will be utilized:

1. K-folds Cross Validation
2. Leave-one-out Cross Validation
3. Parameter tuning with GridSearchCV



## The following is a quick summary of the variables found in the datasets:


Variable Name | Description | 
------------------- | -------------- |
'#' | ID for each pokemon |
Name | Name of each pokemon
Type 1 | Each pokemon has a type, this determines weakness/resistance to attacks
Type 2 | Some pokemon are dual type and have 2
Total | sum of all stats that come after this, a general guide to how strong a pokemon is
HP | hit points, or health, defines how much damage a pokemon can withstand before fainting
Attack | the base modifier for normal attacks (eg. Scratch, Punch)
Defense | the base damage resistance against normal attacks
SP Atk | special attack, the base modifier for special attacks (e.g. fire blast, bubble beam)
SP Def | the base damage resistance against special attacks
Speed | determines which pokemon attacks first each round
Height | height of the pokemon expressed in 100g (so 69 would mean 6.9kg)
Weight | weight of the pokemon expressed in 10cm (so 0.7 would mean 70cm)
base_experience | base experience received when defeating this pokemon
is_default | is an identifier stating this pokemon is the base form in the event there are different versions
    
 
### The datasets used in the model can be found via: 
  
  - https://www.kaggle.com/abcsds/pokemon
  - https://github.com/veekun/pokedex/blob/master/pokedex/data/csv/pokemon.csv


There is a much more clean version for those interested that combines both datasets, that can be found here: https://github.com/DaveRGP/pokRdex/blob/master/pokRdex.csv
