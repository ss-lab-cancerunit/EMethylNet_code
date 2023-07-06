

# A plotting R script produced by the REVIGO server at http://revigo.irb.hr/
# If you found REVIGO useful in your work, please cite the following reference:
# Supek F et al. "REVIGO summarizes and visualizes long lists of Gene Ontology
# terms" PLoS ONE 2011. doi:10.1371/journal.pone.0021800


# --------------------------------------------------------------------------
# If you don't have the ggplot2 package installed, uncomment the following line:
# install.packages( "ggplot2" );
library( ggplot2 );
# --------------------------------------------------------------------------
# If you don't have the scales package installed, uncomment the following line:
# install.packages( "scales" );
library( scales );


# --------------------------------------------------------------------------
# Here is your data from REVIGO. Scroll down for plot configuration options.

revigo.names <- c("term_ID","description","frequency_%","plot_X","plot_Y","plot_size","log10_p_value","uniqueness","dispensability");
revigo.data <- rbind(c("GO:0000003","reproduction", 7.871, 0.237,-0.053, 3.135,-4.6591,1.000,0.000),
c("GO:0000122","negative regulation of transcription from RNA polymerase II promoter", 4.380, 1.583,-3.998, 2.881,-18.5761,0.770,0.000),
c("GO:0002376","immune system process",16.463,-1.057,-0.216, 3.455,-7.7557,0.995,0.000),
c("GO:0008150","biological_process",100.000, 0.357, 0.579, 4.239,-108.4149,1.000,0.000),
c("GO:0008152","metabolic process",64.841,-0.770,-0.051, 4.051,-54.5906,0.998,0.000),
c("GO:0009987","cellular process",90.329,-1.367,-0.722, 4.195,-104.8608,0.999,0.000),
c("GO:0022610","biological adhesion", 8.540,-3.224,-0.040, 3.171,-13.2093,0.995,0.000),
c("GO:0023052","signaling",36.613,-2.337, 0.617, 3.803,-33.1831,0.996,0.000),
c("GO:0032501","multicellular organismal process",41.143,-0.905,-0.448, 3.853,-53.1491,0.997,0.000),
c("GO:0032502","developmental process",33.982,-1.579, 0.185, 3.770,-66.1620,0.996,0.000),
c("GO:0040007","growth", 5.447,-0.503,-0.001, 2.975,-9.9034,0.995,0.000),
c("GO:0040011","locomotion", 9.452,-0.493, 0.066, 3.215,-12.2228,0.995,0.000),
c("GO:0048598","embryonic morphogenesis", 3.301,-8.373, 0.299, 2.758,-21.5446,0.774,0.000),
c("GO:0050896","response to stimulus",49.302,-0.257, 1.506, 3.932,-41.0383,0.997,0.000),
c("GO:0051179","localization",36.018, 0.666, 0.167, 3.795,-29.0940,0.996,0.000),
c("GO:0065007","biological regulation",67.069,-1.515, 0.140, 4.065,-69.1408,0.998,0.000),
c("GO:0071840","cellular component organization or biogenesis",36.532,-3.559, 0.943, 3.802,-31.3251,0.996,0.000),
c("GO:0098742","cell-cell adhesion via plasma-membrane adhesion molecules", 1.344, 0.306, 0.024, 2.369,-8.5032,0.984,0.000),
c("GO:0007154","cell communication",36.705,-0.270,-0.223, 3.804,-32.0807,0.991,0.005),
c("GO:0008283","cell proliferation",11.321,-0.866, 0.503, 3.293,-16.7836,0.978,0.019),
c("GO:0043087","regulation of GTPase activity", 4.126,-1.897,-0.021, 2.855,-5.8852,0.938,0.042),
c("GO:0042592","homeostatic process", 9.371,-3.165, 0.317, 3.211,-7.9711,0.951,0.048),
c("GO:0031344","regulation of cell projection organization", 3.306, 3.433,-3.666, 2.759,-8.0426,0.879,0.061),
c("GO:0030334","regulation of cell migration", 3.912, 4.319, 1.209, 2.832,-6.0956,0.881,0.063),
c("GO:0007169","transmembrane receptor protein tyrosine kinase signaling pathway", 3.976, 1.968, 6.861, 2.839,-5.9454,0.892,0.063),
c("GO:0006928","movement of cell or subcellular component",10.987, 0.143,-0.239, 3.280,-11.7425,0.960,0.064),
c("GO:0008284","positive regulation of cell proliferation", 4.859, 1.667,-0.085, 2.926,-8.9814,0.872,0.065),
c("GO:0006915","apoptotic process",10.537,-0.015,-0.016, 3.262,-9.7950,0.944,0.078),
c("GO:0008219","cell death",11.881,-0.940, 0.183, 3.314,-9.3980,0.960,0.080),
c("GO:0044237","cellular metabolic process",59.486, 1.728,-7.735, 4.013,-58.2509,0.916,0.084),
c("GO:0050794","regulation of cellular process",60.427, 3.394, 3.475, 4.020,-65.8674,0.903,0.094),
c("GO:0070848","response to growth factor", 3.762, 0.588, 6.097, 2.815,-4.8829,0.953,0.110),
c("GO:0065009","regulation of molecular function",17.288, 0.893, 1.437, 3.477,-13.3169,0.945,0.116),
c("GO:0065008","regulation of biological quality",20.202, 1.077, 1.394, 3.544,-19.9228,0.943,0.125),
c("GO:0006793","phosphorus metabolic process",18.702, 1.126,-6.653, 3.511,-20.5052,0.931,0.139),
c("GO:0048880","sensory system development", 0.023,-5.210, 0.152, 0.699,-10.0021,0.891,0.150),
c("GO:0001708","cell fate specification", 0.444,-5.682, 0.219, 1.892,-11.3586,0.861,0.165),
c("GO:0050808","synapse organization", 1.425, 2.957,-3.406, 2.394,-4.4253,0.946,0.173),
c("GO:0051716","cellular response to stimulus",40.358, 1.750, 8.167, 3.845,-34.1945,0.949,0.174),
c("GO:0034330","cell junction organization", 1.466, 3.156,-3.452, 2.407,-7.5262,0.946,0.174),
c("GO:0098727","maintenance of cell number", 0.871,-5.890, 0.243, 2.182,-6.3928,0.882,0.175),
c("GO:0019827","stem cell population maintenance", 0.854,-7.045, 0.249, 2.173,-6.5847,0.873,0.178),
c("GO:0048519","negative regulation of biological process",26.855, 2.429, 3.658, 3.668,-34.6967,0.926,0.183),
c("GO:0023061","signal release", 2.441, 4.014, 4.469, 2.627,-5.3923,0.911,0.188),
c("GO:0046483","heterocycle metabolic process",34.576, 1.016,-7.133, 3.778,-36.8036,0.917,0.193),
c("GO:0051276","chromosome organization", 3.566, 3.527,-3.561, 2.792,-5.5373,0.958,0.197),
c("GO:1901360","organic cyclic compound metabolic process",35.978, 0.479,-6.326, 3.795,-37.4062,0.928,0.198),
c("GO:0051239","regulation of multicellular organismal process",15.268,-5.622, 0.456, 3.423,-29.0079,0.851,0.198),
c("GO:0048518","positive regulation of biological process",30.969, 2.857, 3.841, 3.730,-43.0054,0.923,0.198),
c("GO:0009628","response to abiotic stimulus", 6.705, 0.899, 6.486, 3.066,-4.1965,0.967,0.199),
c("GO:0006725","cellular aromatic compound metabolic process",34.801, 0.864,-6.950, 3.780,-34.4141,0.917,0.199),
c("GO:0009894","regulation of catabolic process", 3.006, 1.096,-2.028, 2.718,-4.3914,0.881,0.199),
c("GO:0009058","biosynthetic process",36.924, 1.062,-7.062, 3.806,-42.7288,0.932,0.201),
c("GO:0071495","cellular response to endogenous stimulus", 7.144, 0.502, 5.749, 3.093,-10.9696,0.966,0.202),
c("GO:0006325","chromatin organization", 4.143, 3.389,-3.500, 2.857,-4.5006,0.959,0.203),
c("GO:0044087","regulation of cellular component biogenesis", 4.784, 2.882,-3.187, 2.919,-4.4955,0.914,0.205),
c("GO:0003006","developmental process involved in reproduction", 3.641,-7.722, 0.310, 2.801,-4.1982,0.855,0.212),
c("GO:0006807","nitrogen compound metabolic process",40.081, 1.712,-7.548, 3.842,-51.5568,0.931,0.212),
c("GO:0048589","developmental growth", 3.272,-7.228, 0.273, 2.754,-10.3187,0.847,0.213),
c("GO:0019222","regulation of metabolic process",35.730, 2.930,-5.619, 3.792,-43.9058,0.891,0.217),
c("GO:0009719","response to endogenous stimulus", 9.175, 0.793, 6.452, 3.202,-9.6835,0.965,0.218),
c("GO:0048736","appendage development", 0.981,-6.317, 0.369, 2.233,-6.5680,0.854,0.219),
c("GO:0060173","limb development", 0.981,-6.976, 0.045, 2.233,-6.5680,0.837,0.219),
c("GO:0060541","respiratory system development", 1.137,-7.940, 0.228, 2.297,-6.4995,0.842,0.223),
c("GO:0033554","cellular response to stress",10.496, 1.259, 6.834, 3.260,-8.3478,0.953,0.227),
c("GO:0043408","regulation of MAPK cascade", 3.883, 3.157, 5.436, 2.829,-6.2956,0.745,0.227),
c("GO:0051674","localization of cell", 8.257, 3.127, 0.869, 3.156,-9.0381,0.967,0.228),
c("GO:0009605","response to external stimulus",12.043, 1.045, 6.857, 3.320,-9.9508,0.963,0.237),
c("GO:0060322","head development", 4.091,-7.041, 0.590, 2.851,-14.8349,0.840,0.239),
c("GO:0003002","regionalization", 1.922,-8.485, 0.403, 2.524,-17.9196,0.838,0.239),
c("GO:0061061","muscle structure development", 3.376,-7.576, 0.492, 2.768,-11.7785,0.844,0.240),
c("GO:0044260","cellular macromolecule metabolic process",48.748, 2.094,-7.473, 3.927,-45.9224,0.884,0.246),
c("GO:0007389","pattern specification process", 2.441,-8.608, 0.364, 2.627,-20.2696,0.837,0.247),
c("GO:0001501","skeletal system development", 2.810,-8.168, 0.352, 2.688,-18.2035,0.823,0.252),
c("GO:0071705","nitrogen compound transport", 4.472, 4.073, 1.074, 2.890,-4.7751,0.967,0.255),
c("GO:0010646","regulation of cell communication",17.651, 2.669, 5.232, 3.486,-28.3162,0.900,0.256),
c("GO:0035295","tube development", 3.283,-7.486, 0.370, 2.756,-19.6877,0.830,0.258),
c("GO:0003008","system process",11.575,-5.894,-0.036, 3.303,-5.3506,0.909,0.259),
c("GO:0072001","renal system development", 1.621,-6.977, 0.386, 2.450,-7.1290,0.831,0.262),
c("GO:0043170","macromolecule metabolic process",53.306, 1.653,-7.629, 3.966,-43.4510,0.917,0.265),
c("GO:0034641","cellular nitrogen compound metabolic process",37.432, 0.978,-6.454, 3.812,-37.5338,0.875,0.267),
c("GO:0001655","urogenital system development", 1.823,-6.821, 0.338, 2.501,-7.7127,0.833,0.267),
c("GO:0050789","regulation of biological process",63.456, 3.127, 3.154, 4.041,-65.5703,0.922,0.274),
c("GO:0043412","macromolecule modification",24.172, 0.791,-5.967, 3.622,-18.1481,0.916,0.275),
c("GO:0006950","response to stress",21.310, 1.048, 6.663, 3.567,-9.9093,0.959,0.292),
c("GO:0048583","regulation of response to stimulus",21.610, 2.493, 7.691, 3.574,-26.6646,0.902,0.294),
c("GO:0050767","regulation of neurogenesis", 3.855,-8.194, 0.357, 2.825,-17.9921,0.699,0.296),
c("GO:0044238","primary metabolic process",60.306, 1.743,-7.765, 4.019,-53.3739,0.919,0.297),
c("GO:0048863","stem cell differentiation", 0.946,-6.205, 0.274, 2.217,-7.7183,0.849,0.297),
c("GO:0007423","sensory organ development", 2.972,-7.298, 0.508, 2.713,-14.4441,0.811,0.298),
c("GO:0001568","blood vessel development", 3.347,-7.707, 0.412, 2.764,-11.5306,0.802,0.304),
c("GO:1901576","organic substance biosynthetic process",36.451, 1.321,-7.258, 3.801,-43.9362,0.866,0.304),
c("GO:0007267","cell-cell signaling", 9.025, 1.536, 5.323, 3.195,-15.4840,0.932,0.306),
c("GO:0042221","response to chemical",23.993, 1.182, 7.475, 3.619,-18.5526,0.958,0.307),
c("GO:0010467","gene expression",30.542, 1.433,-6.502, 3.724,-30.6318,0.909,0.308),
c("GO:0045165","cell fate commitment", 1.402,-6.474, 0.234, 2.387,-17.1266,0.842,0.313),
c("GO:0071704","organic substance metabolic process",62.701, 1.622,-7.403, 4.036,-52.2779,0.917,0.313),
c("GO:0016043","cellular component organization",35.684, 4.016,-3.667, 3.791,-33.8477,0.943,0.314),
c("GO:0031323","regulation of cellular metabolic process",33.837, 4.658,-5.337, 3.768,-46.8379,0.787,0.314),
c("GO:0019538","protein metabolic process",31.991, 1.498,-6.632, 3.744,-18.9858,0.907,0.316),
c("GO:0002520","immune system development", 4.605,-8.500, 0.419, 2.903,-8.5581,0.808,0.319),
c("GO:0048762","mesenchymal cell differentiation", 1.068,-7.547, 0.378, 2.270,-4.7168,0.803,0.338),
c("GO:0007166","cell surface receptor signaling pathway",15.868, 2.553, 7.103, 3.439,-20.5709,0.864,0.355),
c("GO:0030030","cell projection organization", 7.963, 2.392,-2.924, 3.140,-20.2061,0.931,0.359),
c("GO:1901564","organonitrogen compound metabolic process",13.324, 0.145,-5.189, 3.364,-23.9874,0.912,0.365),
c("GO:0001654","eye development", 1.956,-6.952, 0.390, 2.531,-9.7101,0.816,0.366),
c("GO:0007517","muscle organ development", 2.083,-7.281, 0.430, 2.559,-6.8586,0.819,0.369),
c("GO:0048645","animal organ formation", 0.352,-6.636, 0.344, 1.792,-4.4993,0.821,0.376),
c("GO:0048522","positive regulation of cellular process",27.732, 3.869, 4.084, 3.682,-43.4579,0.856,0.376),
c("GO:0048732","gland development", 2.418,-7.143, 0.378, 2.623,-8.1874,0.816,0.376),
c("GO:0035556","intracellular signal transduction",15.101, 2.520, 7.585, 3.418,-20.2246,0.866,0.388),
c("GO:0006810","transport",29.215, 4.875, 1.209, 3.704,-14.7432,0.955,0.389),
c("GO:0048731","system development",25.695,-7.558, 0.298, 3.649,-59.1173,0.763,0.397),
c("GO:0072359","circulatory system development", 5.361,-8.662, 0.457, 2.968,-15.6016,0.806,0.411),
c("GO:0032879","regulation of localization",14.409, 4.865, 1.025, 3.398,-11.0585,0.906,0.412),
c("GO:0008104","protein localization",14.414, 3.969, 0.743, 3.398,-6.4825,0.964,0.413),
c("GO:0009790","embryo development", 5.505,-7.724, 0.476, 2.980,-29.4958,0.817,0.414),
c("GO:0051641","cellular localization",14.905, 3.175, 0.817, 3.412,-7.9289,0.963,0.417),
c("GO:0050793","regulation of developmental process",12.949,-8.079, 0.820, 3.351,-26.9392,0.781,0.419),
c("GO:0051128","regulation of cellular component organization",13.335, 3.307,-2.408, 3.364,-13.5914,0.890,0.420),
c("GO:0048523","negative regulation of cellular process",24.951, 4.231, 4.239, 3.636,-33.8837,0.867,0.426),
c("GO:0010033","response to organic substance",16.515, 1.150, 7.367, 3.457,-17.9925,0.944,0.427),
c("GO:0043933","macromolecular complex subunit organization",14.259, 2.893,-3.213, 3.393,-4.6696,0.950,0.429),
c("GO:0001763","morphogenesis of a branching structure", 1.114,-6.469, 0.368, 2.288,-8.0309,0.825,0.430),
c("GO:0033036","macromolecule localization",16.624, 3.600, 0.927, 3.460,-8.9784,0.963,0.432),
c("GO:0010463","mesenchymal cell proliferation", 0.271, 1.331,-0.754, 1.681,-4.5799,0.974,0.432),
c("GO:0007010","cytoskeleton organization", 6.861, 2.543,-3.002, 3.076,-8.0816,0.954,0.439),
c("GO:0051246","regulation of protein metabolic process",14.651, 2.974,-5.931, 3.405,-12.2519,0.815,0.440),
c("GO:0009888","tissue development",10.912,-7.629, 0.510, 3.277,-22.1513,0.820,0.442),
c("GO:0007165","signal transduction",33.618, 2.788, 6.803, 3.765,-28.2723,0.841,0.449),
c("GO:0023051","regulation of signaling",17.946, 2.738, 5.778, 3.493,-28.1154,0.907,0.454),
c("GO:0071363","cellular response to growth factor stimulus", 3.618, 1.139, 7.224, 2.798,-4.1424,0.944,0.456),
c("GO:0044085","cellular component biogenesis",17.149, 3.225,-3.328, 3.473,-13.0111,0.954,0.458),
c("GO:0046903","secretion", 8.892, 4.221, 1.136, 3.188,-6.5991,0.946,0.463),
c("GO:0035239","tube morphogenesis", 1.973,-7.568, 0.357, 2.535,-19.0981,0.803,0.464),
c("GO:0006811","ion transport", 9.158, 4.722, 1.293, 3.201,-4.2693,0.963,0.466),
c("GO:0007264","small GTPase mediated signal transduction", 2.891, 2.152, 7.110, 2.701,-4.6559,0.895,0.473),
c("GO:0009653","anatomical structure morphogenesis",13.872,-8.158, 0.460, 3.381,-46.2751,0.812,0.474),
c("GO:0034097","response to cytokine", 4.714, 0.573, 6.270, 2.913,-4.6417,0.951,0.475),
c("GO:0010464","regulation of mesenchymal cell proliferation", 0.202, 1.409,-0.776, 1.556,-5.1912,0.940,0.478),
c("GO:0006996","organelle organization",19.411, 2.631,-3.052, 3.527,-16.2714,0.947,0.479),
c("GO:0006464","cellular protein modification process",22.683, 1.037,-6.050, 3.595,-17.3985,0.875,0.485),
c("GO:0048534","hematopoietic or lymphoid organ development", 4.339,-8.586, 0.429, 2.877,-7.2549,0.795,0.485),
c("GO:0048585","negative regulation of response to stimulus", 8.003, 2.550, 6.990, 3.142,-9.2108,0.870,0.491));

one.data <- data.frame(revigo.data);
names(one.data) <- revigo.names;
one.data <- one.data [(one.data$plot_X != "null" & one.data$plot_Y != "null"), ];
one.data$plot_X <- as.numeric( as.character(one.data$plot_X) );
one.data$plot_Y <- as.numeric( as.character(one.data$plot_Y) );
one.data$plot_size <- as.numeric( as.character(one.data$plot_size) );
one.data$log10_p_value <- as.numeric( as.character(one.data$log10_p_value) );
one.data$frequency <- as.numeric( as.character(one.data$frequency) );
one.data$uniqueness <- as.numeric( as.character(one.data$uniqueness) );
one.data$dispensability <- as.numeric( as.character(one.data$dispensability) );
#head(one.data);


# want the size of the points to be the intersection size. We can find this in the gProfiler output:
library(data.table)
gProfiler_data <- fread('/Tank/methylation-patterns-code/methylation-patterns-izzy/biology/gProfiler/gProfiler_.csv', data.table = FALSE)
one.data <- merge(one.data, gProfiler_data[, c('term_id', 'intersection_size')], by.x = 'term_ID', by.y = 'term_id', all.x =TRUE, all.y = FALSE, sort = FALSE)

# now looking at number of parents - we don't want to show terms at the top of the list (ie biological process)
library(GO.db)
parents <- as.list(GOBPPARENTS)

count_num_parents <- function(go_term) {
  p <- parents[[go_term]]
  print(go_term)
  if (p[['isa']] == 'all') {
    return(1)
  }
  else {
    return(1 + count_num_parents(p[['isa']]))
  }
}
# make a column for number of parents in the GO tree
one.data$num_parents <- unlist(lapply(X = as.character(one.data$term_ID), FUN = count_num_parents))


# --------------------------------------------------------------------------
# Names of the axes, sizes of the numbers and letters, names of the columns,
# etc. can be changed below

p1 <- ggplot( data = one.data );
p1 <- p1 + geom_point( aes( plot_X, plot_Y, colour = log10_p_value, size = intersection_size), alpha = I(0.3) ) + scale_size_area();
# p1 <- p1 + scale_colour_gradientn( colours = c("blue", "green", "yellow", "red"), limits = c( min(one.data$log10_p_value), 0) );
p1 <- p1 + scale_colour_gradientn( colours = c('#a50f15', '#de2d26','#fb6a4a', '#ffffb2'), limits = c( min(one.data$log10_p_value), 0) );
p1 <- p1 + geom_point( aes(plot_X, plot_Y, size = intersection_size), shape = 21, fill = "transparent", colour = I (alpha ("black", 0.1) )) + scale_size_area();
# p1 <- p1 + scale_size( range=c(5, 30)) + theme_bw(); # + scale_fill_gradientn(colours = heat_hcl(7), limits = c(-300, 0) );
p1 <- p1 + scale_size( range=c(1, 15), breaks = c(500, 1000)) + theme_bw(); # + scale_fill_gradientn(colours = heat_hcl(7), limits = c(-300, 0) );
# ex <- one.data [ one.data$dispensability < 0.15, ]; 

num_to_label <- 20
library(dplyr)
# ex <- filter(one.data, uniqueness > 0.92) # remove all terms that are not very unique
ex <- filter(one.data, num_parents >= 4) # only keep terms with 4 or more parents (so quite specific terms)
# do 3 or 4 
ex <- arrange(ex, log10_p_value)[1:num_to_label, ]
# ex <- one.data[one.data$log10_p_value < -39.4, ]
dim(ex)


# p1 <- p1 + geom_text( data = ex, aes(plot_X, plot_Y, label = description), colour = I(alpha("black", 0.85)), size = 3 );

library(ggrepel)
p1 <- p1 + geom_label_repel(data = ex, aes(plot_X, plot_Y, label = description, colour = log10_p_value), size = 2.4, fontface = 'bold', force =3, label.padding = 0.1, label.size = 0, box.padding = 0.1, segment.size = 0.2, force_pull=0.5,nudge_y = 1, nudge_x=-2, max.iter = 5000); 

p1 <- p1 + labs (y = "semantic space x", x = "semantic space y");
p1 <- p1 + theme(legend.key = element_blank(), legend.title = element_text(size = 8), legend.text = element_text(size = 8), axis.text=element_text(size=8), axis.title=element_text(size=8), panel.grid.major = element_blank(), panel.grid.minor = element_blank()) ;
one.x_range = max(one.data$plot_X) - min(one.data$plot_X);
one.y_range = max(one.data$plot_Y) - min(one.data$plot_Y);
p1 <- p1 + xlim((min(one.data$plot_X)-one.x_range/10)+1,max(one.data$plot_X)+one.x_range/10);
p1 <- p1 + ylim((min(one.data$plot_Y)-one.y_range/10),(max(one.data$plot_Y)+one.y_range/10) -2);

p1 <- p1 + guides(size = guide_legend(override.aes = list(fill = '#ffffff', alpha = 1, order = 2), title = 'Size of intersection'), colour = guide_colourbar(title = 'Log10 (p value)', order = 1, barheight = 4))

# p1 <- p1 + theme_classic()
# --------------------------------------------------------------------------
# Output the plot to screen

# svg("revigo_plot_multiclass_smaller.svg", width = 5.5, height = 3.5)
# svg("revigo_plot_multiclass_smaller.svg", width = 8, height = 2.5)
svg("/Tank/methylation-patterns-code/methylation-patterns-izzy/biology/gProfiler/revigo_plot_multiclass_smaller_test.svg", width = 8, height = 3.5)
p1;
dev.off()

