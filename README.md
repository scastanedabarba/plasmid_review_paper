# Plasmids, a Molecular Cornerston of Antimicrobial Resistance in the One Health Era

###  **Authors:** Salvador Castaneda-Barba, Eva M. Top, Thibault Stalder

## Link to publication

**********

**Abstract:** Antimicrobial resistance (AMR) poses a significant threat to human health. The widespread prevalence of AMR is largely due to the horizontal transfer of antibiotic resistance genes (ARG), typically mediated by plasmids. Many of the plasmid-mediated resistance genes in pathogens originate from environmental, animal, or human habitats. Despite evidence that plasmids mobilize ARG between these, we have a limited understanding of where ARG emerge and how their ecology and evolutionary trajectories result in them appearing in clinical pathogens. One Health, a holistic framework, enables the exploration of these knowledge gaps. With this in mind, we first provide an overview of how plasmids drive local and global AMR spread and link different habitats. Then we review some of the emerging studies integrating an eco-evolutionary perspective, opening a discussion on factors that affect the ecology and evolution of plasmids in complex microbial communities. Specifically, we discuss how the emergence and persistence of AMR plasmids can be affected by varying selective conditions, spatial structure, environmental heterogeneity, temporal variation, and coexistence with other members of the microbiome. These factors, along with others yet to be investigated, collectively determine the emergence and transfer of plasmid-mediated AMR within and between habitats at the local and global scale.  

**********

## Information about this repository:  

This repository contains the code for the meta-analysis carried out in the review paper titled "Plasmids, a Molecular Cornerstone of Antimicrobial Resistance in the One Health Era".  

### **File Description:**
#### **Data obtained from PLSDB: [plsdb.tsv]() and [plsdb.abr]()**
The files named plsdb.tsv and plsdb.abr were downloaded from PLSDB, version 2021_06_23_v2. The file plsdb.tsv contains the meta-data for all the plasmids while the file plsdb.abr contains the resistance gene annotations. 
#### **Code for meta-analysis: [plasmid_meta.py]()**
The file named plasmid_meta.py contains the script for carrying out the meta-analysis in Figure 3.
#### **Classified plasmid sources table: [classifications.csv]()**
The table classifications.csv contains the habitat to which each plasmid was assigned. The columns 'Host_BIOSAMPLE' and 'IsolationSource_BIOSAMPLE' within plsdb.tsv contain metadata related to the source from which the plasmid was isolated. This information was used to determine whether each plasmid originated from Human, Animal, or Environmental habitats.  
#### **Code for Figure 3: [figure3.py]()**
The file named figure3.py contains the code for generating the figures.
