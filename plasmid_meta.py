import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#import the info table
info = pd.read_csv('plsdb.tsv', sep = '\t')

#only keep the columns that we are interested in for each dataframe
info_mod = info[['ACC_NUCCORE', 'Host_BIOSAMPLE',
                 'IsolationSource_BIOSAMPLE', 'SamplType_BIOSAMPLE']]

'--------------- Classification of plasmids into habitats -------------------'
'''
The metadata available in two columns of 'plsdb.tsv' is used to classify plasmids
into habitats of origin. First, plasmids are classified using the information in 
the Host_BIOSAMPLE column. In cases where the information is missing or unclear, 
plasmids are classified as NA. A second attempt at classification of these plasmids
is carried out, using information from the IsolationSource_BIOSAMPLE column.
'''

#-----Classification using Host_BIOSAMPLE---------
#unique entries in Host_BIOSAMPLE are manually placed into dictionary with appropriate habitat
biosample_class = {'Human': ['Homo sapiens', 'Homo sapiens; female', 'Himo sapiens', 'Homosapiens', 'Homo sapiens (respiratory patient)', 'Homo sapiens (female)', 'female', 'patient', 'homo', 'Human (male, 4.5 yrs)', 
                             'Human gut', 'infants', 'infants', 'human oral', 'Feces, human', 'Healthy human (stool)', 'Human Blood', 'Human Clinical (blood)', 'Human Clinical Sample (fatal pneumonic plague)', 
                             'Human GI tract', 'Human Pleural Fluid', 'Human Stool', 'Human blood', 'Human clinical', 'Human clinical sample', 'Human faeces', 'Human fecal samples', 'Human feces', 
                             'Human intestinal microflora', 'Human septic tank', 'biological fluid-human', 'human', 'human blood', 'human case of meningitis', 'human cell culture', 'human fecal sample', 
                             'human feces', 'human feces (woman, 24 years old)', 'human feces (woman, 60 years old)', 'human feces of different individuals', 'human listeriosis', 'human skin', 
                             'human skin swab', 'human stool', 'Clinical sample, blood', 'clinical', 'clinical isolate', 'clinical patient', 'clinical sample', 'Diarrheal patient', 'Feces of patient with diarrhea', 
                             'Feces of patient with dysentery', 'Feces of patient with hemolytic uremic syndrome', 'From patient with pneumonia', 'From patient with wound infection', 'Sputum (Cystic Fibrosis Patient)', 
                             'blood of a hospitalized patient', 'hospital patients', 'hospitalized patients', 'isolated from blood of a patient with liver abscess and meningitis', 'vomit of food poisoning patient', 
                             'Korean adult feces', 'Korean infant feces', 'adult feces', 'Stool (adult)'],
                   
                   'Animal':['wild chukar', 'pig', 'Turkey', 'Bos taurus', 'duck', 'cattle','fish', 'Sus scrofa domesticus', 'Canine', 'swine', 'Aratinga solstitialis', 'Gallus gallus', 'shrimp', 'Beef', 'Cow', 
                             'Swine', 'Pig', 'Migratory bird', 'Penaeus vannamei', 'Anguilla japonica', 'porcine', 'Goose', 'Rattus rattus', 'Salvelinus fontinalis', 'Sebastes schlegeli', 'Sus scrofa','Gallus gallus domesticus',
                             'piglet', 'pigs','Bivalve mollusk', 'dog', 'nematode', 'Cattle', 'broiler chicken', 'Cat', 'cow', 'cat', 'Rabbit', 'Gallus', 'Paguma larvata' 'Equus caballus', 'goose','Shellfish','Yak',
                             'horse', 'Dog', 'Partridge', 'Quail', 'poultry', 'Oryctes gigas', 'Canis lupus familiaris', 'Felis catus', 'Horse', 'Mercenaria mercenaria', 'Anodonta arcaeformis', 'Giant Panda',
                             'canine', 'Equus ferus caballus', 'dairy cattle', 'bird', 'Rat', 'Turbot', 'cage-cultured red drum', 'Bovine', 'Paralichthys olivaceus', 'Chicken', 'Farmed turkey', 'Procambus clarkii',
                             'pheasant', 'partridge', 'Farmed pheasant', 'Broiler chicken', 'Chroicocephalus novaehollandiae (Australian Silver Gull chick)', 'bovine', 'Duck', 'slaughtered pig', 'the intestine membrane of a diarrheic piglet', 'the respiratory tract of a pig with swine respiratory disease',
                             'pigeon', 'Cows', 'Ovis aries', 'Pork', 'Rainbow trout', 'Perinereis linea', 'Oreochromis niloticus', 'Hydrophilus acuminatus', 'pet dog', 'rabbit', 'catfish', 'Crocodylus siamensis', 'Crow',
                             'Atlantic salmon', 'egret', 'Porcine', 'Giant panda', 'Moschus berezovskii','Bos mutus', 'Bos taurus coreanae (Korean native cattle)', 'wild bird', 'Salmo salar', 'Equus caballus', 
                             'Paguma larvata', 'chicken', 'Dairy cow', 'Lacertilia', 'Nezara viridula (Cotton Pathogen Vector; southern green stink bug)', 'Goat', 'Sparrow', 'Marmota baibacina','Bos primigenius taurus','Sebastes schlegelii',
                             'Papio papio','Odocoileus virginianus','Phoca largha','crow','Marmota sibirica', 'Corvus brachyrhynchos','Seriola dumerili','Pelodiscus sinensis','Sablefish','mouse', 'Macrobrachium nipponense',
                             'Mus musculus', 'Rattus norvegicus', 'Spermophilus sp.','Gull','Dermacentor andersoni','Urocitellus undulatus','Canis latrans','yak','Neopsylla setosa','Junco','Hawk','Pteropus livingstonii', 'Pteropus poliocephalus',
                             'cormorant', 'Deer','Hippopotamus amphibius','sheep','gull', 'Ondatra zibethicus', 'Argopecten purpuratus', 'Mus musculus subsp. domesticus', 'Neophocaena asiaeorientalis', 'ruddy shelduck','Galleria mellonella', 'Farmed red-legged partridge', 'Bactrocera dorsalis', 'Ixodes pacificus', 
                             'calanoid copepod', 'Penaeus japonicus', 'Amblyomma variegatum (cattle tick)', 'Gut of wasp', 'Dicentrarchus labrax','Blattella germanica', 'Pseudorca crassidens', 'Allacta bimaculata', 'Ictalurus punctatus', 'Tuberolachnus salignus', 
                             'Tegillarca granosa', 'plateau pika', 'Diaphorina citri', 'Ischnocodia annulus', 'Ixodes ricinus', 'silkworm', 'Anas platyrhynchos', 'Ixodes persulcatus', 'Melanaphis sacchari', 'Odontobutis platycephala',
                             'Chelymorpha alternans', 'Blatta orientalis', 'Paralichthys olivaceus (flounder)', 'Sitobion avenae', 'seabass','oyster', 'Meleagris gallopavo', 'Tibetan antelope', 'Pediculus humanus corporis', 'Nauphoeta cinerea', 
                             'Draeculacephala minerva', 'Cassida viridis', 'Apsterostigma', 'Halyomorpha halys', 'Haemaphysalis juxtakochi', 'Cistudinella sp.', 'Mya arenaria oonogai Makiyama', 'Nasonia vitripennis', 'Morone chrysops x Morone saxatilis', 
                             'Marmota himalayana', 'Seriola lalandi', 'Apostichopus japonicus', 'Crocodile lizard', 'Mustela putorius furo', 'Mastotermes darwiniensis', 'Trichodesmium erythraeum IMS101', 'Pthirus gorillae', 'Caenorhabditis briggsae',
                             'Solea senegalensis','Drosophila melanogaster Oregon-R modENCODE','Cinara cedri','Mus musculus C57Bl/6J','squirrel','crucian carp','Omphisa fuscidentalis Hampson','Cyclopterus lumpus','Perca fluviatilis',
                             'Ornithodoros hermsi','Apterostigma dentigerum','Silurus asotus','bat','Trichonympha agilis','Bat','equine','Apis mellifera','Parachirida sp.','Japanese rhinoceros beetle larva','Oecophylla smaragdina',
                             'Aphis craccivora','Bos taurus coreanae','Macroplea mutica','Chaeturichthys stigmatias','Acromis sparsa','Neoaliturus tenellus','Glossina brevipalpis','Bombyx mori','mouse-C57Bl/6J','Termite','Acyrthosiphon kondoi',
                             'white stork','Equus kiang','Acyrthosiphon pisum','Cassida sp.','Megacopta punctatissima','Physopelta gutta','Trypoxylus dichotomus','Amazona sp.','Oncorhynchus kisutch','Charidotella sexpunctata','Gentoo penguin',
                             'Oyster larvae','Macaca silenus','Togo hemipterus','insect','Drosophila melanogaster','Tanakia koreensis','Crassostrea gigas (Pacific oyster)','Skeletonema marinoi strain RO5AC','caprine','Cerambycidae sp.',
                             'Ixodes ovatus','Shrimps','Plateumaris pusilla','Phyllophaga sp.','Riptortus pedestris','Marmot','Parrot','Penaeus vannamei (shrimp)','parakeet','Pentalonia nigronervosa','Ixodes pacificus (western blackleg tick)',
                             'Cassida vibex','Cryptocercus punctulatus','Pseudotrichonympha sp.','Cryptocercus clevelandi','Peromyscus leucopus','tortoise','Blaberus giganteus','Ellychnia corrusca','Amblyomma neumanni','toothfish',
                             'Geronticus eremita','Polytelis','Ixodes scapularis','Uroleucon ambrosiae','Macrotermes barneyi','Argas persicus','Apterostigma','invertebrates','Trachinotus ovatus','Columba livia','Amblyomma cajennense',
                             'Hyadaphis tataricae','Camponotus chromaiodes','Ixodes stilesi collected from Oligoryzomys longicaudatus','goat','Ailuropoda melanoleuca','Drosophila melanogaster Oregon-R-modENCODE','Lagenorhynchus acutus',
                             'Caenorhabditis elegans','Glossina morsitans morsitans','Nusuttodinium aeruginosum','turkey','Bankia setacea','Marmota','Oryctolagus cuniculus','Trimyema compressum','Discus','Alexandrium minutum','Phoca vitulina',
                             'Rattus','Sus scrofa domestica','Agroiconota sp.','Himantormia','Macrosiphum gaurae','Crassostrea gigas','Penaeus (Litopenaeus) vannamei (whiteleg shrimp)','Channa argus (snakehead fish)','Callyspongia','wax moth',
                             'Sipalinus gigas','Vultur gryphus','Skeletonema marinoi strain ST54','Bursaphelenchus xylophilus','Pyropia','Discomorpha sp.','Ovis aries (domestic sheep)','American cockroach','Lion-tailed macaques', 'Locusta migratoria',
                             'beluga whale','Oreochromis','Apodemus agrarius','Delphinapterus leucas (Beluga whale)','alligator','Panulirus ornatus','Aphis helianthi', 'Chicken feces', 'Cow feces', 'Chroicocephalus novaehollandiae',
                             'fulmars','Mink','Helicobacter pylori','Baizongia pistaciae','honeybees',"Pyrus communis 'Williams'",'Oncorhynchus mykiss','turtle','Penaeus monodon', 'cloacae swab- farm', 'Euphausia superba','Scophthalmus maximus', 'Scylla serrata',
                             'Cattle Hide', 'Cattle faeces', 'Cattle feces', 'Cattle slaughter plant', 'animal-cattle-steer', 'beef from cattle Peranakan Ongole', 'cattle feces', 'cattle or sheep', 'cattle slurry', 'Oedothorax gibbosus', 'Pan troglodytes verus (Western chimpanzee)', 
                             'Psittacus erithacus feces', 'Swiss alpine Ibex feces', 'Turkey feces', 'a feces sample of chicken origin', 'a feces sample of migratory birds origin', 'a feces sample of swine origin', 'Paper wasp', 'Penaeus vannamei (Whiteleg shrimp)',
                             'Pig faecal swab', 'Pig rectal swab', 'Wild pig, fecal', 'intestine membrane of a diarrheic piglet', 'pig fecal', 'pig feces', 'Swine Final Chilled Carcass', 'a anal swab sample of swine origin', 'a nose swab sample of swine origin', 
                             'animal-swine-sow', 'swine cecum', 'swine nasal swab', "swine's gut", 'tissue and/or biological fluid (swine)', 'Aphis fabae', 'Avian', 'Bigeye tuna', 'Bighead Carp', 'Buteo jamaicensis', 'Cancer pagurus', 'Canis lupus', 'Canis lupus bichon frise', 
                             'Canis lupus labrador retriever', 'Canis lupus mixed', 'Canis lupus pekingnese', 'Ceratitis capitata', 'Haliotis discus hannai', 'Heterostera chilensis', 'Mizuhopecten yessoensis', 'Mouse', 'Muscaphis stroyani', 
                             'Nebria ingens riversi', 'Spodoptera frugiperda', 'Sus', 'Sus scrofa scrofa', 'Trichechus manatus', 'Trigona sp.', 'blue sheep', 'calf', 'grass carp', 'hen','mare', 'marmot', 'red kangaroo', 'tick'],
                   
                   
                   'Env.':['sludge', 'flower', 'Environment', 'rice','date palm', 'Alhagi sparsifolia Shap.', 'Adenophora trachelioides Maxim.', 'Sewage', 'Actinidia chinensis', 'Lichen', 
                                  'Capsicum annuum', 'Sakura tree', 'Oryza sativa', 'Rhizoma kaempferiae','Actinidia deliciosa','Citrus aurantiifolia','Pear','Trifolium spumosum L. (Annual Mediterranean clovers)','Botryllus sp.',
                                  'Mimosa affinis','masson pine','olive knot (Olea europaea)','Cryptocercus kyebangensis','Vicia sativa','Lettuce','Sesame','sesame seedling','Hibiscus','Psoroma sp.','Schizaphis graminum biotype I',
                                  'Gossypium hirsutum','Geodia barretti','Populus davidiana x Populus alba var. pyramidalis (PdPap)','Rubus sp.','Stereocaulon sp.','Prunus avium','Cotinus coggygria','Myzus persicae (green peach aphid)',
                                  'Pepper','Cowpea','Euonymus sp.','Tribulus terrestris','eucalyptus','Chinese cabbage','Aphis glycines','Apple','Arachis hypogaea','Medicago sativa L. subsp. ambigua','Plantain','Solanum lycopersicum cv. Brillante',
                                  'Zingiber officinale','lichen Stereocaulon sp.','Veronaeopsis simplex Y34','Lactuca sativa','plantain','Oryza glumipatula','Ficus religiosa L.','Brassica rapa subsp. pekinensis','soybean','Lebeckia ambigua',
                                  'Anguilla anguilla','Mimosa flocculosa','Flammulina filiformis','Plant','red alga','Dracaena sanderiana','Olea europaea','Panax ginseng','Solanum commersonii','Triticum aestivum','Sporobolus anglicus',
                                  'Vitis vinifera','Aphis urticata','Catharanthus roseus','Broussonetia papyrifera','Allamanda cathartica','Citrus sinensis','Androsace koso-poljanskii','grape','Prunus cerasifera','Trifolium pratense',
                                  'Punica granatum','Medicago truncatula','Shinkaia crosnieri','Pisum sativum l. (pea)','bacteria','winter wheat','Camellia oleifera','Panicum miliaceum','Biserrula pelecinus L.','Tomato',
                                  'Brassica rapa subsp. chinensis','Nipponaphis monzeni','Brittle root','Triticum aestivum L','Areca catechu','Rhodosorus marinus','Prosopis cineraria','citrus spp.','Glycine soja','Allium cepa',
                                  'Rhodobacter sphaeroides 2.4.1','Antho dichotoma','Hyacinthus orientalis','Usnea','Tephrosia purpurea subsp. apollinea','muscari','Bacillus thuringiensis subsp','Plum',
                                  'Naegleria','Lotus sp.','G.max','Grapevine','plant','Glycine max cv. Jinju1','Solanum lycopersicum (tomato)','Rosa','mung bean','Sorghum bicolor','Avena sativa',
                                  'Citrus','pear','Stachytarpheta glabra','citrus','eggplant','Fava bean','Cryptomeria japonica var. sinensis','Allium cepa (onion)','Solanum tuberosum','Oxytropis kamtschatica','Pyrus pyrifolia','Melilotus officinalis',
                                  'Medicago sativa','Zophobas atratus','Agave americana L.','Lemna minor','Zea mays','Commiphora wightii','Glycine max','Pisum sativum','Camellia sinensis','Cucurbita maxima','mandarin orange','Grapefruit',
                                  'Oxytropis triphylla','Capsicum frutescens','Leiosporoceros dussii','Beta vulgaris','Malus prunifolia (crab apple)','Achillea ptarmica','Trifolium repens','soil','tomato','pickled cabbage',
                                  'Phaseolus','Haliclona simulans','Vicia faba','turfgrass','e','Citrus x paradisi','Solanum lycopersicum','Elymus tsukushiensis','Phaeoceros','Myzus persicae','Phaseolus sp.','Chrysanthemum x morifolium',
                                  'Robinia pseudoacacia','alfalfa','Lathyrus sativus','Blasia pusilla', 'Parthenium argentatum Gray (guayule shrubs)','Lotus','Vicia alpestris','Echinacea purpurea','wild cotton',
                                  'Arabidopsis','garlic, cabbage','Vachellia farnesiana','sugarcane','Casuarina equisetifolia','Lactuca sativa L. var. longifolia','kimchi cabbage','soybean (Glycine max)','Onion','Pyropia tenera',
                                  'Acaciella angustissima','Salvia splendens','Glycine max L. Merr.','Curcuma aromatica','Wheat','Fusarium oxysporum f. sp. cucumerinum','Onobrychis viciifolia','Cucumis sativus','eucalypti of Eucalyptus',
                                  'Citrus limon','weed','Eruca vesicaria subsp. sativa','Actinidia','Lutjanus guttatus (Rose snapper)','peanut','Triticum aestivum L.','Leontopodium alpinum','Prunus sp.','common bean','Calliandra grandiflora',
                                  'Ulva pertusa (algae)','Carrot','Lolium arundinaceum','Cladonia borealis','Solanum melongena','Zea mays L.','Juglans regia','Miscanthus x giganteus','Brassica juncea var. foliosa','Hirudo verbana','Cicer arietinum',
                                  'Capsicum sp.','pepper plant','Lotus corniculatus',"Vitis vinifera cv. 'Izsaki Sarfeher'",'Rice','Pseudotrichonympha grassii (protist) in the gut of the termite Coptotermes formosanus','Mimosa',
                                  'Leontopodium nivale','Musa sp.','Oxytropis pumilio','Vavilovia formosa','Pyrus pyrifolia var. culta', 'Eucommia ulmoides','ginger','Thlaspi arvense','koumiss','Tanacetum vulgare','Medicago arborea','Tobacco',
                                  'Aphis craccivora (cowpea aphid)','Digitaria eriantha','Syngnathus typhle','pepper','Potato', 'Astragalus pelecinus','Persea americana','Microlophium carnosum','cabbage','sweet orange',
                                  'tomato rhizosphere','tobacco','Lespedeza cuneata','Phaseolus vulgaris','Citrus hystrix', 'Medicago orbicularis','wheat','Soybean (Glycine max (L.) Merrill)','Pyropia yezoensis conchocelis',
                                  'Environmental (Pond)', 'Estuary water environment', 'Hospital environment', 'Hypersaline environment', 'Nosocomial environment', 'PIF Production Facility Environment', 'Pig environment', 'Plastic debris in land/lake environment', 
                                  'Poultry environment', 'anaerobic environments', 'bakery environment - assembly production room', 'bakery environment - hallway', 'bakery environment concentrated whipped topping', 'brewery environment', 'environment', 'environment of small animal veterinary clinic',
                                  'environment swab', 'environmental', 'environmental sample', 'environmental surface', 'environmental swab', 'environmental; wetland', 'in the nicotine environment', 'marine environment', 'marine hydrothermal environment', 'sewage from Environmental Centre Robert O. Picard', 
                                  'winery environment', 'Activated sludge of a municipal wastewater treatment plant in Klosterneuburg, Austria.', 'Breeding wastewater', 'E-waste recycling site', 'Freshwater sample from downstream of wastewater treatment plant', 
                                  'Freshwater sample from upstream of wastewater treatment plant', 'Hospital waste water', 'Medical waste water', 'River sediment contaminated by e-waste', 'Solid waste landfill sample', 
                                  'Wastewater effluent sample', 'Wastewater influent sample', 'Wastewater treatment of pharmaceutical company', 'Wastewater treatment plant', 'activated sludge from pharmaceutical waste in Shaoxing China', 
                                  'activated sludge of a wastewater treatment facility', 'activated sludge of wastewater treatment plants', 'electronic waste-contaminated sediment', 'isolated from activated sludge collected at the Gold Bar wastewater treatment facility', 
                                  'sludge from bioreactor treating oxytetracycline bearing wastewater', 'sludge of membrane bioreactor in a waste-water treatment plant', 'the activated sludge of Harbin Taiping wastewater treatment plant', 'waste sludge of a brewery plant', 'waste water', 
                                  'waste water and activated sludge', 'waste water treatment system', 'wastewater', 'wastewater from pig manure', 'wastewater treatment plant', 'Hospital sewage', 'Sewage water', 'Taihu New City Sewage Treatment Plant', 'Treated sewage effluent', 
                                  'anodic biofilm of glucose-fed microbial fuel cell originally inoculated with sewage sludge', 'hospital sewage', 'raw sewage', 'sewage', 'sewage & soil', 'sewage plant sludge', 'sewage sludge', 'sewage tank', 'sewage treatment plant', 'sewage water', 'sewage water sludge', 
                                  'Biocathode MCL, marine biofilm', 'Marine Biofilm','Non-filtered water from the water column of tank 6 of a marine aquarium containing stony-coral fragments. Water maintained at 26 degree C', 
                                  'Surface of a polyethylene microplastic particle present in tank 6 of a marine aquarium containing stony-coral fragments and water maintained at 26 degree C', 
                                  'Surface of a sandy sediment particle present in tank 6 of a marine aquarium containing stony-coral fragments and water maintained at 26 degree C', 'Malus sieversii', 'Medicago lupulina L.', 'Medicago sativa subsp. varia (Martyn) Arcang', 'Megaspora',
                                  'barial marine waters', 'marine', 'marine macroalga Fucus spiralis', 'marine mud', 'marine seawater', 'marine sediment', 'marine sediments', 'Banana', 'Phyla canescens', 'Pogostemon cablin', 'Prunus dulcis', 'Raspberry', 'Rosa sp.',
                                  'marine sediments in Liaodong Bay', 'marine sludge', 'marine sponge', 'A bacterial consortium from cattle pasture soil, USA', 'Alpine forest soil', 'Antarctic soil', 
                                  'Coal mine soil', 'Contaminated Mine Soils', 'Dichlobenil-treated soil sampled from the courtyard of a former plant nursery located above a BAM-contaminated aquifer near Hvidovre', 
                                  'Hexachlorocyclohexane (HCH) pesticide-contaminated soil', 'Libyan oil-polluted soil', 'Lycium barbarum rhizosphere soil', 'Microscale soil grain', 'Nan Madol, Pohnpei soil', 
                                  'Organic soil', 'Oryza sativa, rhizosphere soil', 'PCBs contaminated soil', 'Paddy soil under the long-term application of triazophos insecticide', 'Polluted soil', 
                                  'Potato rhizosphere, Soil', 'Rice field soil', 'Rice-wheat rotation soil', 'Saline Desert Soil', 'Sandy soil near a stream', 'Soil (sea shore)', 'Soil cores containing cogon grass roots', 
                                  'Soil cores containing the root system of one maize plant', 'Soil sample from China, Shenyang', 'Thai soil', 'agricultural soil', 'arsenic-contaminated soil', 'bamboo forest soil', 
                                  'carbendazim contaminated soil', 'coal mine soil', 'contaminated soil', 'contaminated soil sample obtained from a drilling mud pit', 'crude oil-contamination soil', 
                                  'date palm rhizospheric soil', 'desert soil', 'dyed contaminated landsoil', 'estuarine soil', 'field soil', 'forest soil', 'forest soil of Shengnongjia', 'Urochloa reptans', 'Vigna radiata',
                                  'forest soil, soft coal slag', 'fresh creek bank soil, topsoil', 'frozen soil', 'fruit tree rhizosphere soil', 'garden soil', 'gold and copper mining soil', 'grass soil', 
                                  'grassland soil', 'heavy metal containmented soil', 'heavy metal polluted soil', 'landfill cover soil', 'linuron-contaminated soil', 'long-term organic manure fertilized alkaline soils', 
                                  'maize rhizosphere soil', 'mangrove soil', 'meropenem treated soil', 'muddy soil from torrent', 'nitrile contaminated soil', 'oil contaninated soil', 'oil-polluted soil', 'paddy soil', 
                                  'paddy soil contaminated with arsenic due to geogenic reasons', 'paddy soils', 'petroleum-contaminated soil', 'petroleum-contaminated soil with high salinity', 
                                  'petroleum-contaminated soils', 'podzolic fallow soil sampled from the ARRIAM experimental field', 'Dust collector of pigpen', 'pig feed from feed plant',
                                  'purple rhizosphere soil the cabbage Brassica campestris L. ssp. chinensis Makino (var. communis Tsen et Lee)', 'rhizosphere of apple plantlets grown in replant diseased soil', 
                                  'rhizosphere soil', 'rhizosphere soil of tomato plants', 'rhizospheric soil', 'rice paddy soil', 'riparian wetland soil', 'river bed soil', 'saline soil', 'Skeletonema costatum',
                                  'silty clay loam soil (pH 6.1)', 'soda solonchak soil', 'soil core', 'soil from Chengdu', 'soil from a small paddock', 'Allium cepa L.', 'Angelica sinensis Dlies',
                                  'soil from courtyard of a former plant nursery located above a BAM-contaminated aquifer', 'soil near hot water effluent', 'soil of Hengshui Lake', 'soil sample', 
                                  'soil sample from Harissa region in Lebanon', 'soil sample from dry pine forest', 'soil sample, Enrichment culture with 4-chlorophenol and 2,4-dichlorophenol', 
                                  'soil sample, enrichment culture methanol as carbon source', 'soil under chromium-containing slag heap', 'tea plantation soil', 'tomato rhizosphere soil', 'uncultivated field soil', 
                                  'Pooled cattle faecal samples collected from floor of farm', 'Pooled pig faecal samples collected from floor of farm', 'Pooled sheep faecal samples collected from floor of farm', 
                                  'Beach water', 'Biofilm - potable water', 'Cooling tower water', 'Deep seawater from Mariana Trench', 'Filtered Seawater, 0.2-1 micron fraction', 'Fresh water', 'Hot water tap, Geest Office Building', 
                                  'Industrial starch water', 'Jin River water W2', 'Jin River water W5', 'La Roche-Posay thermal water', 'New Delhi seepage water sample', 'Nile Delta Mediterranean Sea surface water', 
                                  'North Sea (surface water)', 'Ocean water', 'Sea water', 'Seawater collected from a tide pool', 'Seawater recirculating aquaculture system biofilter', 'Clematis', 'Cotton',
                                  'Sediment of a freshwater pond enriched in a fixed-bed reactor with 2,6-dichlorophenol as sole carbon and energy source', 'Stream water', 'Surface seawater', 'catalpa',
                                  'Surface seawater of South China Sea', 'Water (artesian well)', 'Water collected from a pond', 'Water for algae observation', 'Water from pond', 'Water from the baltic sea', 
                                  'Water kimchi', 'Water of rice field', 'Water sample', 'Zimmer water bag', 'acidic mine water', 'aquarium water', 'bottom water', 'condensation water of the Shenzhou-10 spacecraft', 
                                  'creek water', 'deep seawater', 'drinking water', 'ex cyanobacteria field sample from freshwater', 'fresh water', 'freshwater', 'Blueberry', 'Brassica rapa',
                                  'freshwater lake cyanobacterial bloom; sample filtered onto 1.2 um poresize filter', 'freshwater mud', 'glacial stream water', 'groundwater', 'groundwater from well FWB306-02', 
                                  'heated-cooler unit water tank', 'lake water', 'meltwater pond', 'oil-contaminated water', 'oil-production water', 'oilfield produced water', 'open ocean water', 'polar sea water', 
                                  'pond water', 'product water in oilfield', 'river surface water', 'river water', 'riverwater', 'sea water', 'seawater', 'shallow tropical waters, normally from coral reef substrate', 
                                  'stream surface water', 'stream water', 'stream/river surface water', 'sulfidic waters (60 m) from the Peruvian upwelling region', 'surface sea water', 'surface seawater', 'surface water', 
                                  'surface water of Lake Taihu', 'surface water of the southern North Sea', 'the condensate water of the Shenzhou-9 spacecraft', 'water', 'water column of an alpine lake', 'Lolium perenne', 'Lotus japonicus',
                                  'water from surface of lake', 'water gallon', 'water kefir', 'water of air-conditioning systems', 'water pond', 'water rainwater-tap Botanic Garden Hamburg', 'water stream', 
                                  'water surface of oligotrophic pond', 'biofilm boat', 'biofilm reactor', 'biofilm sample', 'microbial mat/biofilm', 'rock biofilm and bottom sediments', 'stream biofilm', 
                                  'Black sea sediments', 'Deep sea sediment', 'Glacial lake sediment', 'North Atlantic Rise deep-sea sediment', 'North Sea (sediment)', 'Riverbed sediment', 'Sandy intertidal sediment', 
                                  'Seashore sediment', 'Sediment around river', 'Sediment top', 'Soudan mine sediment', 'Stream sediment', 'Tidal flat sediment sample', 'West sea sediment in South Korea', 'coastal sediment', 
                                  'creek sediment', 'deep sea sediment', 'deep-sea sediment', 'glacier sediment', 'lake sediment', 'mangrove sediment', 'mangrove sediments', 'mine sediment', 'mud sediment', 'ocean sediment', 
                                  'oil-contaminated sediment', 'pond sediment', 'river sediment', 'sediment', 'sediment collected at a cold seep field', 'sediment from seaside wetland around Yalujiang river', 'Solanum muricatum',
                                  'sediment in spartina alterniflora', 'sediment of a eutrophic reservoir', 'sediment sample from the Liaodong Bay of the Bohai Sea', 'sediment stream', 'sedimentation pond in a zinc factory', 
                                  'stream sediment', 'subseafloor sediment', 'Manure compost', 'chicken manure', 'parent strain CV601 collected from dairy manure', 'pig manure', 'Agricultural field', 'Sichuan pickle', 'Skeletonema menzelii', 'Solanum lycopersicum L.', 
                                  'Agricultural field (no beans grown in the 5 previous years)', 'Paddy field, Sungai Manik, Malaysia', 'Rice Fields', 'Shengli Oilfield', 'high-temperature oilfield', 'Ficus benjamina', 'Fragariae ananassa', 'Grateloupia sp.',
                                  'Contaminated sludge', 'Foam on activated sludge', 'Isolated from activated-sludge plants', 'activated sludge', 'activated sludge in our lab', 'active sludge', 'active sludge around pharmaceutical factory', 
                                  'food sludge compost', 'reactor sludge', 'sludge of an anaerobic digestion reactor', 'Hospital effluents', 'alcohol foam dispenser in hospital intensive care unit', 'bedside light switch in hospital intensive care unit', 
                                  'bedside rail in hospital intensive care unit', 'hospital', 'hospital sink', 'sink handle in ICU Room in Military Hospital', 'swab from a hand-washing sink as part of the hospital routine surveillance program', 
                                  'washroom sink in hospital intensive care unit', 'Broiler chicken farm', 'Turbot fish farm', 'bioaerosol of a chicken farm', 'drag swab of litter-farm', 'farm kitchen', 'open pond on an algae farm', 
                                  'organic chicken farm', 'Chinese fermented food-pickles', 'Fermented Meat Product','Fermented corn meal', 'Fermented dairy products', 'Fermented liquor based on wild grass', 'Fermented onions', 'Fermented pork (nham)', 
                                  'Fermented sausage, Norway', 'Fermented shrimp paste', 'Fermented vegetables', 'Fu-Tsai (Fermented vegetable, food)', 'Jogaejeotgal, a Traditional Korean Fermented Seafood', 'Korean fermented food', 'Meju, fermented soybean', 
                                  'Myeolchi-jeotgal, salted fermented food', 'Naturally fermented tofu whey', 'Saeu-jeotgal, salted fermented food', 'Traditional fermented dairy products in Inner Mongolia', 'Traditional fermented food in Korea', 
                                  'Traditional fermented soybean pastes', 'Vietnamese fermented sausage (nem chua)', 'Yak fermented milk', 'fermented cabbage', 'fermented cabbage (Food)', 'fermented cassava roots (fufu)', 'fermented chinese cabbage', 'fermented fish', 'fermented food', 
                                  'fermented lentils', 'fermented milk', 'fermented salami', 'fermented vegetable', 'fermented vegetables', 'fermented whole fish product suan yu', 'kimchi (Korean traditional fermented food)', 'kimchi (traditional fermented korean dish)', 
                                  'natually fermented tofu whey', 'raw fermented sausage', 'the fermented dough in Dali', 'traditional Thai fermented pork sausage', 'Curcuma wenyujin Y.H. Chen et C. Ling', 'Dongxiang wild rice', 'Erigeron annuus L. Pers',
                                  'Himalayan blackberry', 'Ipomoea aquatic', 'Japanese radish', 'Kandelia candel (mangrove)', 'Kimchi','Leptinotarsa decemlineata', 'Neopyropia tenera', 'Nostoc flagelliforme',
                                  'lentil', 'meat duck', 'mulberry', 'potato', 'sofa'],
                   
                   np.nan: ['nan', 'missing', 'Missing', 'unknown', 'not applicable', 'not collected', 'Unknown', 'Not Available', 
                            'Glycine max cv. AC Glengarry', 'seujeot', 'none', 'Not applicable', 'Not Applicable', '-', np.nan], 
                   
                   'unclear': ['pneumoniae', 'NIST Mixed Microbial RM strain', 'Vectobac','UCC strain', 'Acanthamoeba', 'Skate', 'Acanthamoeba polyphaga HN-3', 'protozoa', 'Canicola', 'Cereibacter sphaeroides', 'Cereibacter sphaeroides 2.4.1', 
                               'Glycine', 'Hainan', 'Homo', 'Musa balbisiana cultivar Kepok','Seth Lab strain', 'Shanxi','Sichuan', 'Turbo', 'Yunnan', 'jamiecosley','male']}

#use reversed dictionary to append the class to the dataframe
reversed_class = {val: key for key in biosample_class for val in biosample_class[key]}
info_mod.loc[:,'sample_class'] = info_mod['Host_BIOSAMPLE'].map(reversed_class)

#fill in the null values with info from the isolationsource
info_mod.loc[:,'Host_BIOSAMPLE'].fillna(info_mod['IsolationSource_BIOSAMPLE'], inplace=True)

#dictionary for classifying the new entries into the biosample collumn that came from isolationsource
biosample_class2 = {'Human': ['Woman with cystitis', 'human', 'Human fecal samples', 'hospital patients', 'Blood of 26-month child with malaria and anemia','Clinical sample, blood','Healthy human (stool)', 'From patient with pneumonia', 'clinical patient', 'clinical', 
                              'clinical isolate', 'biological fluid-human', 'hospitalized patients','human blood','From patient with wound infection', 'blood of a hospitalized patient','Infant diarrheic stool', 'Feces, human', 'vomit of food poisoning patient', 'Urine from hospitalized patient',
                              'central_venous_catheter','infant sputum','Clinical: Human (Homo sapiens)','tracheal secretion isolated from an uncertain age woman','From patient with blood stream infection','Human_no contact with swine','urine from healthy adult with cystitis','Burn Patient',
                              'HA-MRSA blood clinical site','purulent absess from military trainee','rectal swab from healthy college student','human feces','cerebrospinal fluid, hospital','Clinical sample: urine','Diarrheal patient','bile from holelithiasis patient','Homo',
                              'clinical urine culture','Purulent absess from military trainee',"separated from patient's sputum",'blood culture from maile','Human stool sample','bronchial sample from a male patient with chronic obstructive pneumonia','clinical sample','human stool',
                              'genital tract of a healthy woman', 'healthy infant fecal samples', 'stool (infant)', 'infant feces'],
                   
                   'Animal':['Chicken faeces', 'Manis javanica', 'cows', 'White-lipped Deer','Black-collared Starling', 'pork', 'Veal', 'Pooled pig faecal samples collected from floor of farm','monkey kidney tissue-culture fluids of the FH strain (Eaton Agent Virus)', 'gill tissue of Bathymodiolus japonicus', 'Sheep placental tissue',
                             'Pooled cattle faecal samples collected from floor of farm', 'Pooled sheep faecal samples collected from floor of farm', 'Pig rectal swab', 'diarrheal snake','healthy broiler chicken','duck with tremor', 'wild yak feces', 'chicken', 'Psittacus erithacus', 'Siniperca scherzeri',
                             'swine', 'goat', 'Konosirus punctatus', 'Meleagris gallopavo','swine nasal swab', 'Dog with mastitis', 'goose anus swab','Pig faecal swab', 'bovine mastitis', 'mouse gut',  'Turkey feces', 'trach wash', 'Calf Liver','Chicken Cecal Content', 'scallop', 'Black-headed gull',
                             'Neophocaena phocaenoides','Swine Final Chilled Carcass', 'slaughtered pig','pig feces', 'Chicken tissue', 'Mastitis Milk','chicken carcass', 'bovine lymph node','young chicken', 'a feces sample of migratory birds origin', 'cow dung', 'iguana', 'Mussels', 'Musca domestica',
                             'chicken liver', 'chickens', 'goose faeces', 'Wild boar','a feces sample of swine origin','intestine membrane of a diarrheic piglet','Turkey with septicemic infection','Housefly', 'chicken manure','the intestine membrane of a diarrheic piglet', 'raw cow milk','White-crowned sparrow',
                             'Chicken cecum','swine cecum','pig manure','Broiler chicken farm', 'broiler chick cecum', 'animal-swine-sow', 'animal-cattle-steer', 'Diseased Labeo rohita fish','Bovine pre-evisceration carcass at harvest','Heifers','turkey','a anal swab sample of swine origin', 
                             'chicken skin','a feces sample of chicken origin','Chicken feces', 'Bird of prey','a nose swab sample of swine origin','laying hen withcolibacillosis', 'separated from the corpses of silkworms that had died due to Bb natural infection in Daiyue district, Taian city, Shandong province, China.', 
                             'tissue and/or biological fluid (swine)', 'Poultry carcass', 'organic chicken farm', 'Bearded dragon','Bovine (bobby calf, fecal)', 'Sheep fecal sample','cattle feces', 'Larvae from hard clams (Mercenaria mercenaria)','sick pig','cow feces','feces from pig', 'loach',
                             'diseased flounder','cloacal swab','Muscaphis stroyani','Chicken product','cecal contents of broiler chicken','Farmed partridge','leg bone from commercial broiler','Healthy pig from farm','Bovine Organ','bovine pre-evisceration carcass','Bovine adult', 'Equine Body Fluid/Excretion',
                             'Animal hide', 'Animal, carcass of dead zebra', 'Animal faecal matter', 'Tissue, animal', 'animal', 'Rectal swab from healthy swiss fattening veal calves', 'Rectal fecal grab samples from a commercial feedlot', 'turbot', 'Chamaeleonidae', 'Snake','unidentified actinians',
                             'chicken trachea', 'Post-chill chicken carcass', 'Chicken Liver', 'retail chicken gizzard', 'Chicken fecal', 'ground chicken', 'retail chicken liver', 'chicken meat and conveyer belt', 'chicken meat inner strip', 'Qingyuan chicken', 'chicken-abbatoir-Sporadic', 
                             'chicken dung', 'chicken tissues', 'market chicken', 'chicken caecal content', 'Retail Chicken',  'Fecal samples from slaughtered sheep', 'gill of flatfish', 'Canine oral plaque', 'African Elephants', 'silkworm feces', 'Aphis craccivora (cowpea aphid) on Robinia pseudoacacia (locust)', 
                             'Cow dung', 'Cow rumen', 'silkworm feces', 'Waxworms gut', 'mealworm', 'parrot', 'Mus musculus-9', 'Red-breasted Parakeet', 'panda', 'south China tiger', 'Peltigera membranacea thallus', 'Aborted bovine fetus, brain', 'bovine rumen', 'Bovine carcass', 'Bovine Faeces', 'Calf', 
                             'California Ground Squirrel (Spermophilus beecheyi)','Heifer vaginal mucus', 'Goat, fecal', 'Bison', 'Frog', 'opossum', 'Rabbit, caecal', 'Alpaca, fecal', 'quail', 'Lung of aborted horse fetus', 'Ixodes spinipalpis','Mactra veneriformis', 
                             'Dendrolimus sibericus','Pacific plankton'],
                   
                   
                   'Env.':['Waste water treatment plant', 'Wastewater influent sample','Wastewater effluent sample','Freshwater sample from downstream of wastewater treatment plant', 'Shower 3', 'feed additive imported from China', 
                                  'Freshwater sample from upstream of wastewater treatment plant','Environment','sink drain','cucumber','hand rail', 'river', 'bakery environment - hallway',  'Sewage', 'activated sludge',
                                  'hospital sewage', 'riverwater', 'river water', 'leaf vegetable','Sediment around river','permafrost, Kolyma lowland', 'telephone of Nurse station','denitrifying, sulfide-oxidizing effluent-treatment plant',
                                  'Biofilm - potable water', 'marine sediment', 'Hospital effluents', 'condensation water of the Shenzhou-10 spacecraft','New Delhi seepage water sample', 'surface water', 'Rice Fields', 'fresh water', 'environment', 'fish aquaculture', 
                                  'Water from pond', 'wastewater from pig manure', 'Wastewater treatment plant', 'wastewater', 'Jin River water W2', 'wastewater treatment plant', 'fresh alfalfa sprouts', 'Taihu New City Sewage Treatment Plant', 'sediment', 
                                  'Jin River water W5', 'wet market', 'Laboratory contaminant', 'bakery environment concentrated whipped topping', 'Wet market', 'sludge from bioreactor treating oxytetracycline bearing wastewater', 'Burns unit surveillance', 
                                  'Shower 2', 'water', 'fish larvae aquaculture','environmental; wetland', 'Room 7','algae aquaculture','environmental','active sludge around pharmaceutical factory', 'sink aerator', 'hospital','Hospital sewage','aquarium water',
                                  'Hospital waste water','sea water','bakery environment - assembly production room', 'sediment collected at a cold seep field', 'ocean sediment', 'air','sewage water', 'sewage','The RAS Atlantic Salmon facility', 'soil', 'Chinese pickle',
                                  'Rice-wheat rotation soil','Chicken feed in trough', 'clam larvae aquaculture','ventilator','hospital sink', 'shelf', 'Shelled Pistachios', 'Dust collector of pigpen', 'ARJO bathroom', 'biofilm reactor', 'sewage from Environmental Centre Robert O. Picard',
                                  'Medical waste water', 'rhizosphere','creek sediment','Raw almonds', 'Poultry environment','Creek', 'Raw Almonds','marine environment','Nosocomial environment','slaughterhouse', 'almond kernel (raw, variety carmel)',
                                  'Chicken meat', 'ground turkey', 'chicken breast', 'Chicken breasts', 'Chicken wings', 'Chicken thighs', 'Animal feed', 'Recirculating aquaculture system', 'clam aquaculture', 'zooplankton aquaculture', 'rotifer aquaculture', 'cleaning system aquaculture',
                                  'Galega officinalis root nodule', 'plant roots', 'noduls from roots of Medicago sativa', 'root nodules', 'Galega orientalis root nodule', 'UCB-1 pistachio rootstock', 'Pisum sativum root-nodule', 'roots of peanut', 'Canola Roots', 
                                  'root nodule of Trifolium uniflorum collected on the edge of a valley', 'sugarcane root', 'the root of rice', 'roots', 'root of Codonopsis pilosula', 'plant root', 'Angelica gigas Nakai root surface', 
                                  'root nodule of Sesbania cannabina', 'Halophyte Rhizosphere', 'rhizosphere of Abies nordmanniana', 'Chilli rhizosphere', 'plant rhizosphere', 'Rhizosphere of Soybean', 'peony rhizosphere', 'Pea rhizosphere', 'rhizosphere taken during the low tide period', 
                                  'rice rhizosphere', 'cucumber rhizosphere', 'Maize rhizosphere', 'sugar beet rhizosphere', 'rhizosphere of potato','leaf', 'strawberry leaf tissue', 'leaf tissues of rice in Korea', 'plant leaf', 
                                  'Sour cherry (Prunus cerasus) symptomatic leaf', 'fermentation bioreactor', 'olive fermentation', 'Dairy fermentation', 'Baijiu fermentation starter', 'Dairy Fermentation', 'Apple juice in fermentation (cider)', 'the fermentation starter of Baijiu', 
                                  'The fermentation starter of Baijiu','bioreactor', 'Hygromycin B antibiotic bottle', 'Biomedical source', 'Woodchip bioreactor', 'probiotic preparation "Lactobacterinum"', 'Quinoline-degrading denitrifying bioreactor', 'Integrative Microbiology Research Center, South China Agricultural University (SCAU), Tianhe District, Guangzhou China', 
                                  'probiotics', 'Tibicos symbiotic community', 'Biocathode MCL', 'power plant biotrickling filter', 'Glomma River', 'River', 'from the mud of the eutrophic river Ryck','Korean soybean paste', 'Nuruk, Korean traditional beverage starter', 'Korean kefir', 'Korean traditional alcoholic beverage', 'Flower of Forsythia koreana', 
                                  'Makgeolli (Korean traditional alcoholic beverage)', 'fresh-cut produce processing plant', 'coke and gas plant treatment facilities', 'rubber production plant territory', 'rotted fruit from symptomatic plant', 'Wool from Pakistan', 'ready to eat mixed salad leaves (obtained from discount store)', 'isolated from the deepest Ocean', 'ZC4 compost from a compost operation of Sao Paulo Zoo', 
                                  'moist arsenopyrite (FeAsS)-containing rock taken from a mine tunnel approximately 300 m below the ground in the Granites gold mine', 'swab from a follow-up assessment of the tap handles and sink edges after a first disinfection attempt', 'acid mine decant and tailings from uranium mine', 
                                  'contaminant from air', 'plastics debris from sea coast', 'Thiodendron" bacterial sulfur mat from mineral sulfide spring', 'Euglena gracilis from city ponds', 'isolated from the tobacco substrate', 'from type material', 'SCOBY from Kombucha tea', 'glacier from Lahual Spiti valley', 'microbial mats from Zloty Stok gold mine', 
                                  'derived from strain s4454', 'apple juice from cider press', 'Inner tissues of halophyte Limonium sinense (Girard) Kuntze', 'tidal marsh', 'salt marsh', 'shellfish hatchery', 'fried eel (fish)', 'Shellfish Hatchery', 'marinated fish product', 'Fish ball',
                                  'Beach sand', 'beach sand', 'meromictic lake', 'saline lake', 'cyanobacterial mat sample of Antarctic lake', 'salt lake', 'Anderson Lake', 'Lake Untersee', 'Brazilian saline-alkaline lake', 'lake', 'soda lake Magadi', 'Potato flake', 'North Park Lake', 'Sphagnum peat bog lake', 'Lake Kulim, Kedah, Malaysia', 'Lake Mainaki silt',
                                  'Antarctica: Sea off King George Island', 'shallow sea, symbosis with harmful algal bloom algae', 'Antarctic sea ice', 'sea mud', 'deep-sea hydrothermal vent', 'subseafloor basaltic crust', 'Red Sea lagoons-Mangrove Mud', 'shallow-sea hydrothermal system', 'EMS/AHPND-diseased hepatopancreas', 'deep sea',
                                  'tidal flat sample', 'Weathered rock sample',  'mixed sand sample', 'Carrot juice', 'carrot', 'coral', 'pork steak', 'Bobby veal steak', 'dirt', 'Tianshan glacier', 'metal sulfide rock', 'wheat beer', 'beer', 'bottled beer', 'pilsner beer', 'Beer Keg', 'light wheat beer', 'beer contaminant', 
                                  'intestinal contents of termite Nasutitermes nigriceps', 'Rice wine rice syrup', 'rice seeds', 'Rice seed', 'rice shoot', 'tomato pulp', 'tomato, internal','air of cow shed', 'ocean', 'Ocean', 'Mud', 'mud flat', 'pit mud', 'mangrove mud', 'mud', 'polluted mud', 'black mud', 'the pit mud of a Chinese flavor liquor-making factory', 
                                  'the pit mud of a Chinese liquor factory', 'flour', 'Flour', 'Permafrost', 'ermafrost region of Qilian Mountains', 'Alkaline pool submerged anode electrode', 'Tidal pool', 'Base of single tree in sidewalk along Rollins Rd. Evidence of many dogs, Boston, Mass', 'Flower of Shittah tree', 'Sand', 'oil sands', 'banana', 'fruit fly', 
                                  'canker of kiwifruit', 'fruit', 'hot spring', 'spring', 'acidic hot spring', 'thermal spring', 'compost', 'hyperthermophilic compost', 'Mushroom compost', 'enriched culture of compost', 'Salad', 'lettuce', 'cave', 'Chronically low temperature and dry polar region', 'sugar thick juice', 'degraded sugar thick juice', 'variety pack luncheon meat turkey and ham', 
                                  'Litopenaeus vannamei purchased at the supermarket', 'Pork at market', 'agricultural settling lagoon', 'Paper surface', 'Surface', 'surface of Pikea pinnata', 'brewery-associated surface', 'surface', 'Deep subsurface fluids', 'Crab Marinated in Soy Sauce', 'Spoiled apple cider', 'pineapple', 
                                  'Apple cider', 'stairway step at Antarctic Concordia station', 'pine forest', 'cream products', 'pork product', 'probitic products', 'pasteurized spray dried egg products blend (whole eggs, corn syrup, and salt)', 'mangrove wetland ecosystem', 'almond drupe', 'Orange', 'Campbell Early grape', 'Flower of Rhododendron schlippenbachii', 
                                  'Flower of Rhododendron sclippenbachii', 'Storage tank', 'pond', 'Pond', 'shrimp pond', 'rot potato tubers', 'Mashed potatoes', 'Hydroponic pots with potatoes', 'Concrete','Geothermal Isolate', 'cherry', 'lowbush blueberries', 'blue berry', 'spiced meat', 'meat processing facility', 'Meat retail', 'Raw duck meat', 'lamb meat', 'sprouts', 
                                  'Alfalfa sprouts', 'algae', 'meat processing facility', 'Pickled Green Chili Peppers', 'wilting pepper (C. annuum) stems in Sanya', 'Instant Soup', 'Dried Tofu', 'brine of stinky tofu', 'Stinky Tofu', 'Salar de Atacama, Atacama Desert', 'Sichuan pickle vegetables', 'mustard pickles', 'chinese pickle', 'Lotus corniculatus nodule', 'corn', 
                                  'glacial stream', 'tundra wetland', 'wasp honeycombs', 'sterile body site', 'chemical manufacturing sites', 'Dust', 'potash salt dump', 'palm wine', 'Palm wine', 'wine', 'hydrothermal vent polychaetes', 'hydrothermal vent', 'vinegar factory',
                                  'barnacle at wood pile-on', 'Anaerobic digester', 'Sichuan Red Original Yak Yogurt', 'Diced lamb', 'Tattoo ink', 'stinky xiancaigeng', 'Soybean paste (Chonggugjang)', 'Doenjang(Soybean paste)', 'shrimp paste', 'gold-copper mine', 'underground horizons of a flooded mine in Russia', 'antimony mine', 'wolfram mine tailing', 'Rotten eggs', 
                                  'broken egg shells- hatchery', 'Knife', 'honey', 'Bee Honey', 'White-rot fungus Phanerochaete chrysosporium', 'combined sewer', 'Coastal', 'Oryza sativa, grain', 'combined sewer effluent', 'Cyanobacterial bloom', 'cold seep', 'brewery yeast', 'Raw mutton', 'Air conditioning system', 'Koumiss', 'green chile ingredient', 'Malus sp.', 'palm brown sugar',
                                  'soybeans', 'sourdough', 'Ham', 'gladiolus', 'litchi pericarp', 'Lava', 'AC condensate', 'Coal Bed', 'salami', 'weathered tuff', 'Baengnokdam Summit Crater Area, Mt. Halla', 'local geographical source', 'bed sheets', 'RoundUp', 'monitor panel', 'Chilean kraft-pulp mill effluents', 'cold-stored modified atmosphere packaged broiler filet strips with the first signs of spoilage', 
                                  'fertilizer', 'Origanum marjorana', 'mead', 'Vinegar', 'Runway 10 Reef (10-12 m)', 'tung meal', 'Ogi (red sorghum)', 'Malus sylvestris', 'Saratov Oil Refinery', 'Doenjang', 'yogurts', 'Kefir', 'rye silage', 'vinegar', 'Algal phycosphere', 'montane grasslands', 'contaminated transfused red blood cell unit', 'Jeotgal', 'Coffee Cup', 'oil contaminated tidal flat', 
                                  'Turban shell', 'saltpan', 'lichen', 'Broiler', 'Chives', 'Antarctic iceberg', 'Kombucha SCOBY',
                                  
                                  
                                  'Chicken Breasts', 'Chicken Wings', 'Mixed parts', 'Chicken Legs','Ground Beef', 'Pork Chops bone-in', 'liver and spleen of chicken (broiler)','Pork chops', 'Pork Chops', 'Buffalo milk','chicken drumstick',
                                  'pork & cabbage dumplings', 'duck meat', 'Pork &Cabbage Dumplings', 'beef burger', 'Raw beef', 'retail chicken','retail beef liver', 'Chicken liver', 'meat', 'chicken meat', 'turkey meat', 'Bulk pig ears', 'Cheonggukjang',
                                  'Chicken heart and liver', 'Chicken Meat','Retail chichen', 'minced meat','retail turkey','comminuted chicken','milk','steamed conch','poultry meat', 'retail raw meat','Irradiated ground pork and beef', 'dairy', 'Dairy',
                                  'beef liver','Fermented Meat Product','Instant pork', 'Food (meat)', 'chicken Gizzard', 'food/cold -smoke salmon','product egg raw white', 'product egg raw whole','Raw milk cheese','bovine hide','Pig product','milk from cattle farm',
                                  'beef ground','retail meat','market','gizzard', 'food/cold -smoke salmon', 'Chinese food lobster sauce', 'imported food: soy sprout', 'food', 'imported food: chilli pepper', 'Food (meat)', 'Plant derived food stuff', 'vomit from a food poisoning case', 
                                  'food product', 'traditional greek kasseri cheese', 'cheese', '10 weeks old 45+ Samso cheese', 'Cheese product', 'Kopanisti cheese', 'New Zealand cheese', 'Tilsit cheese', 'Cheddar cheese factory', 'Natural whey culture from Gruyere cheese', 
                                  '10 weeks old Samso 45+ cheese', 'cheese starter culture', 'Soja milk', 'milk from female', 'milk powder production facility', 'yak milk', 'Milk powder', 'Milk fan', 'milk products', "Milker's hand", 'cow raw milk', 'milk after contact with traditional sicilian wooden vat (Tina)', 
                                  'Dried milk','French dry-type pork sausage', 'beef sausage link', 'Chinese sausages', 'raw sausage', 'French dry-type Pork sausage', 'White Kimchi (Baek Kimchi)', 'Diced Radish Kimchi', 'Baechu-kimchi', 'Kimchi without red pepper powder', 'kimchi', 'Galchi-Kimchi', 
                                  'young radish kimchi','Ground beef', 'comminuted beef', 'Ripening beef', 'Beef carpaccio', 'vacuum packed beef', 'ground beef', 'Beef Ribs', 'Beef Deli Slices', 'beef', 'dairy product', 'Tibetan traditional dairy products'],
                    
                   
                   'unclear': ['urine', 'feces','stool', 'blood', 'blood specimen', 'Wound infection','pheasant duodenum','Ensuite 7/8', 'placental tissue', 'Uninoculated HEP-2 tissue culture',
                               'fresh intestinal feces', 'rectum', 'gut','high concentration of fluoride', 'anus swab', 'Mixed salads', 'brain tissue','cecum','Doubanjiang', 'wound', 'throat/groin', 'tracheal aspirate', 'Vein blood',  'pus','faecal sample','wound swab sample',
                               'groin','Gimbap','rectal swab','cell culture', 'perirectal swab', 'Fjerkrae', 'sputum', 'fecal sample','blood sample', 'brewing yeast sample','Blood Culture','wound sample','diarrheal snake diarrheal snake in Hunan','urine sample', 'Korean fermented food',
                               'raw milk cheese', 'Sf9 cell culture media','nasal perirectal swab', 'food','culture mutant', 'CVP line blood','Urethral discharge','purulent abscess','Perineum','urine catheter','Urine sample','Skin ulcer','bile','cecal sample',
                               'cerebrospinal fluid','sperm','Stool/Rectal Swab','decubitus swab','muscle abscess tissue','Peripheral blood','wound swab','skin','excreted bodily substance','sptum','skin swab','deep wounds and lesions','eye', 'gills',
                               'bronchoalveolar lavage','faecal swab','abdominal drainage fluid','Femural surgical wound','Sun Yat-sen Memorial Hospital','Ascites fluid','traumatic discharge','joint fluid','Perirectal swab','CA-MRSA blood site','Wound secretion',
                               'Blood culture','screening swab','Wound swab','abscess','puncture fluid','Perirectal','UCC isolate','Foodborne disease surveillance','pus swab (trachea)','skin and mucous membranes','cloacal swabs','blie', 'valley', 'Whey culture', 'Anabaena culture',
                               'lung','abdominal drain fluid','Urine/Genitourinary','trachea','anqing','Stools','endotracheal tube','synovial fluid','blood-culture','secretion','chaohu','Groin','rectal screen','heart blood','rectal_swab', 'particulate matter', 'Bay of Bengal', 'respiratory tract',
                               'Pus','nose','fecal swab','Trachaeal secretion','Bite Wound','tracheal secretion','Slaughterhouse','Tracheal secretion','urinary tract','induction culture','Peritoneal dialysis fluid','intestinal content','Abdominal Abscess','Oral cavity',
                               'infection','liver','Bronchial fluid','Bronchoalveolar lavage from right lung','intestine','Surveillance','tissue','rectal','hefei','Rectal','Greece','Rectal swab','anal swab','respiratory','anus swab sample','Anal swab','CVC','Anal',
                               'Rectum','nares','peripheral blood','intraabdominal abscess','granulation tissue','Pleural Effusion','bodily fluid','Respiratory sample - sputum','BAL',"Yunsong Yu's Lab",'Brain abscess fluid','upper respiratory tract','abdominal drainage',
                               'fece','nasopharynx','Open pus','Tissue','bood','bronchoalveolar lavage fluid','Transtracheal aspirate','mycoplasma culture contaminant','blood culture','kidney','Allium cepa L.','Not collected','unavailable', 'enrichment cultures from UCC Lynda-Stan', 
                               "derived from CIP 106327 (Collection de l'Institute Pasteur, Paris, France)",'DSMZ culture collection', 'Microcystis culture', 'culture maintained in Leon, missing plasmid pSCL3', 'bacterial culture', 'culture collection', 
                               'Laboratory isolate', 'laboratory stock', 'A laboratory derivative of ATCC 14028s', 'laboratory', 'laboratory strain', 'Poland, Warsaw area', 'Greensboro, Alabama', 'Industrial', 'industrial building air scrubber liquid', 
                               'Kyoto','not isolated', 'DSMZ Isolate', 'LAB', 'lab', 'lab strain', 'butter starter', 'lactic starter (for yogurt making)', 'starter culture', 'urinary tract infection', 'Jiangsu', 'nutrient broth', 'broth', 'Jeonju-si',
                               'China: Zibo, Shandong', 'NA: to be reported later', 'China: Zibo, Shandong', 'ATCC', 'ATCC strain', 'NEB433', 'sterile', 'Tianjin', 'Sungai Pinang, Penang, Malaysia', 'Air sacs','kabura-zushi', 'Tibetan kefir', 
                               'morning glory','whole body','bean blight', 'NCIMB', 'breast abscess','brain', 'Peter Dedon MIT', 'oral', 'bronchial lavage', 'throat', 'R2A medium','Western North Pacific Station S1','saliva', 'KAIST', 'Oropharynx', 'IAM12617',
                               'cyanobacterial aggregates', 'culture of Roseovarius sp. TM1035', 'Gulf of Finland', 'futsai', 'Synechocystis sp. GT-L', 'Tsoundzou', 'Beijing', 'mushroom substrate', 'Nuruk', 'Emmental', 'wheat germ',
                               'Gochujang', "host's whole body", 'Stool', 'Ragi', 'vaginal secretions', 'Natto','nodule','Saeng-gimol meju', 'ascites', 'supragingival plaque, periodontitis', 'Protoplast breeding', 'Minnesota', 'R.J.Roberts',
                               'inflamed gingiva', 'bacterial consortium','carcass', 'Borehole HDN1, spa', 'larvae', 'Freeze-dried stock', 'unkown', 'agar plate','Maotai Daqu', 'Ginseng','conjugation assay', 'skin abscess', 'cucumber "Kurazh F1"', 'obscured', 'Yeongdeuk-gun']}


#create new dictionary that contains all of the keys from sorting through isolation source and host biosample
reversed_class2 = {val: key for key in biosample_class2 for val in biosample_class2[key]}        
combined_reversed_dict = {**reversed_class, **reversed_class2}

#classify all entries based on the combined dictinary
info_mod.loc[:,'sample_class'] = info_mod['Host_BIOSAMPLE'].map(combined_reversed_dict)

#replace values that say unclear with np.nan
info_mod.replace({'unclear': np.nan}, inplace=True)

#drop unclassified plasmids
df_nonan = info_mod[info_mod['sample_class'].notna()]
df_nonan.drop(['IsolationSource_BIOSAMPLE', 'Host_BIOSAMPLE'], axis=1, inplace=True)

"""
------------------------------------------- Generation of data for figure ------------------------------------
"""
#import ARG annotations table and only keep ARG from card with 100% identity and cov
annotations = pd.read_csv('plsdb.abr', sep = '\t')
annotations = annotations[(annotations['pident']==100)&(annotations['cov']==100)]
annotations = annotations[['qseqid', 'sseqid', 'sseqdb']]
annotations_db = annotations[annotations['sseqdb']=='card']
annotations_db.rename(columns = {'qseqid':'ACC_NUCCORE'}, inplace=True)



#-------------------------------Panel A
#merge arg and classification info, separate into plasmids carrying arg and not carrying arg
plasmids = info_mod.merge(annotations_db, on = 'ACC_NUCCORE', how ='outer') #this contains all plasmids
res_plasmids = plasmids[~plasmids['sseqid'].isna()]
nores_plasmids = plasmids[plasmids['sseqid'].isna()]

#combine independent ARG into one row for the res_plasmids. This ensures that each plasmid has one entry with all the arg, not one entry for each arg
split = res_plasmids['sseqid'].str.split(',', n=2, expand=True)
res_plasmids['gene'] = split[0]
res_plasmids['confers_to'] = split[2]
res_plasmids = res_plasmids.groupby('ACC_NUCCORE')['gene'].agg(['unique'])
res_plasmids = res_plasmids.merge(info_mod, on = 'ACC_NUCCORE', how ='inner')
res_plasmids.drop(['IsolationSource_BIOSAMPLE','Host_BIOSAMPLE', 'SamplType_BIOSAMPLE'], axis=1, inplace=True )
nores_plasmids.drop(['IsolationSource_BIOSAMPLE','Host_BIOSAMPLE', 'SamplType_BIOSAMPLE', 'sseqid','sseqdb'], axis=1, inplace=True )

#generate count for each combination, combine into df (df2)
human_nores = nores_plasmids['sample_class'].value_counts()['Human']
human_res = res_plasmids['sample_class'].value_counts()['Human']
animal_nores = nores_plasmids['sample_class'].value_counts()['Animal']
animal_res = res_plasmids['sample_class'].value_counts()['Animal'] 
env_nores = nores_plasmids['sample_class'].value_counts()['Env.']
env_res = res_plasmids['sample_class'].value_counts()['Env.']
d2 = {'class':['Human', 'Human', 'Animal', 'Animal', 'Env.', 'Env.'], 
      'Plasmid ARG Carriage': ['Carries ARG', 'No ARG', 'Carries ARG', 'No ARG', 'Carries ARG', 'No ARG'], 
      'count':[human_res, human_nores, animal_res, animal_nores, env_res, env_nores]}
df2 = pd.DataFrame(data=d2)

#--------------------------------Panel B
res_plasmids2 = df_nonan.merge(annotations_db, on = 'ACC_NUCCORE') #note that plasmids that don't carry ARG will not be in the merged dataframe

#split the sseqid column into resistance gene and what it conferrs resistance to, add these as columns to the df
split = res_plasmids2['sseqid'].str.split(',', n=2, expand=True)
res_plasmids2['gene'] = split[0]
res_plasmids2['confers_to'] = split[2]
split2 = res_plasmids2['gene'].str.split('_', n=1, expand=True)
res_plasmids2['gene'] = split2[0]

#for each arg, look at the habitats in which it is found and the #of plasmids it appears on
#this is reflected in the dataframe below where categories shows tha habitats and count tha number of times it appears
dataframe = pd.DataFrame(columns=['Categories', 'Count'])
for res_gene in (list(res_plasmids2.gene.unique())):

    #create new dataframe of the plasmids which carry the gene of interst
    res_plasmids = res_plasmids2.loc[res_plasmids2['gene'] == res_gene]
    res_plasmids.reset_index(inplace=True)
    count = len(res_plasmids)
    
    #check how many categories the gene is found in
    categories = list(res_plasmids.sample_class.unique())
    categories.sort()
    categories = '\n'.join(categories)

    #append count and categories to df
    dataframe = dataframe.append({'Categories':str(categories), 'Count':count}, ignore_index=True)

category_counts = dataframe.groupby('Categories').size().sort_values(ascending=False).to_frame()
category_counts = category_counts.rename(columns = {0:'counts'})

#--------------------------------Panel C
res_genes = ['OXA-1', 'KPC-1', 'NDM-1', 'NDM-5', 'MCR-1', 'MCR-9', 'MCR-8', 'MCR-3', 'vanXA', 'vanYA', 'vanZA', 'vanHA', 'CTX-M-15', 'TEM-1', 'SHV-12', 'CTX-M-65']

#store number of plasmids classified to each category
counts = info_mod['sample_class'].value_counts().rename_axis('environment').reset_index(name='count')
human = counts.iloc[0, 1]
environment = counts.iloc[1, 1]
animal = counts.iloc[2, 1]

#empty dataframe to which I will append data
percentages = pd.DataFrame(columns=['Gene', 'Percentage', 'Habitat'])

#create new dataframe that contains all connections to be made
for res_gene in res_genes:

    #create new dataframe of the plasmids which carry the gene of interst
    res_plasmids = res_plasmids2[res_plasmids2.gene.str.contains(res_gene, case=False)]
    res_plasmids.reset_index(inplace=True)
  
    #count how many plasmids within each category carry the gene and store that in variables
    plas_counts = res_plasmids['sample_class'].value_counts().rename_axis('environment')
    for index in list(plas_counts.index.values):
        if index == 'Human':
            human_percent = (list(plas_counts.loc[['Human']])[0]/human)*100
            percentages = percentages.append({'Gene':res_gene, 'Percentage':human_percent, 'Habitat':'Human'}, ignore_index=True)
        if index == 'Env.':
            environment_percent = (list(plas_counts.loc[['Env.']])[0]/environment)*100
            percentages = percentages.append({'Gene':res_gene, 'Percentage':environment_percent, 'Habitat':'Env.'}, ignore_index=True)
        if index == 'Animal':
            animal_percent = (list(plas_counts.loc[['Animal']])[0]/animal)*100
            percentages = percentages.append({'Gene':res_gene, 'Percentage':animal_percent, 'Habitat':'Animal'}, ignore_index=True)

#--------------------------------Full figure
fig = plt.figure(tight_layout=True, figsize=(6.38, 5))
gs = fig.add_gridspec(100,100)
sns.set_style("ticks")   
plt.rcParams['font.size'] = '8'

ax1 = fig.add_subplot(gs[0:40,0:26])
ax1 = sns.histplot(df2, x='class', weights='count', hue='Plasmid ARG Carriage', multiple='stack', edgecolor='white', shrink=0.8)
#ax1.set_title('Plasmids')
ax1.set_xlabel('Habitat', fontweight='bold')
ax1.set_ylabel('Plasmid Count', fontweight='bold')
ax1.set_ylim([0, df2.groupby(['class']).sum()['count'].max()*1.15])
plt.legend(labels=['No ARG', 'Carries ARG'], loc = 'upper right')    

ax2 = fig.add_subplot(gs[0:40, 38:])
ax2 = sns.histplot(category_counts, x='Categories', weights='counts', edgecolor='white', color = 'green', shrink=0.8)
ax2.set_ylabel('ARG Count', fontweight='bold')
ax2.set_ylim([0, category_counts['counts'].max()*1.15])
ax2.set_xlabel('Habitat Combinations', fontweight='bold')


ax3 = fig.add_subplot(gs[60:96,:])
ax3 = sns.stripplot(x='Gene', y='Percentage', hue='Habitat', size = 5, data = percentages, jitter=.2)
plt.legend(loc='upper center', title='Habitat')
ax3.set_ylabel('Percentage of Plasmids', fontweight='bold')
ax3.axes.xaxis.set_visible(False)

ax4 = fig.add_subplot(gs[95:,:])
height = [1]*len(res_genes)
bars = res_genes
x_pos = np.arange(len(bars))
color = []
for gene in res_genes:
    counts = percentages['Gene'].value_counts().to_frame()
    gene_count = counts.loc[gene,'Gene']
    if gene_count == 3:
        color.append('red')
    if gene_count == 2:
        color.append('black')
    if gene_count == 1:
        color.append('white')
ax4 = plt.bar(bars, height, color=color, align='center', width = .9)
plt.yticks([])
plt.xticks(rotation=45)
plt.margins(x=.0166)

plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
plt.margins(0,0)
#plt.savefig('meta_analysis.pdf')