best_prompt = {
    'airplane': [' The nose is the front part of the aircraft that houses the cockpit.',
                 'There is no wing part of an airplane in this grayscale map.',
                 ' The back end of an airplane, with its two engines and two methods of steering.',
                 'This sentence is describing a partial airplane that is being shown in a depth map.'],
    'bag': ['The bag has a black strap that goes over the shoulder.',
            ' Ready for Literally AnythingThis bag is perfect for carrying all of your essentials with you on the go.'],
    'cap': [
        'Assuming you are talking about a baseball cap, the crown is typically the highest point of the hat, and the panels are the pieces of fabric that make up the sides and back of the hat.',
        'This sentence is about the peak value of the grayscale depth map.'],
    'car': ['the car has a metal roof that is slanted down towards the back.',
            'The hood of a car is an important part of the vehicle.', 'I need new tires for my car.',
            'The engine is the heart of a car.'],
    'chair': ['Back of the chair depth map.', ' A closeup of the seat pad of a chair.',
              'A leg part of a chair 3D model can look like a cylinder, a square, or a rectangle.',
              'In a chair depth map, an armrest would appear as a horizontal line at the appropriate depth.'],
    'earphone': ['This is the earcup of a earphone in a three-dimensional map.',
                 'This is the headband part of an earphone in a 3D map.',
                 'A typical earphone has a wire that consists of four parts: the inner conductor, the dielectric insulation, the outer conductor, and the jacket.'],
    'guitar': [
        'This sentence is saying that the head or tuning pegs are the only part of the guitar that is shown in the depth map.',
        'It is a representation of a gray 3D guitar model.',
        'The body part of a guitar can be identified in this grayscale map by its shape.'],
    'knife': [
        'A depth map of a knife typically shows the blade as a thin, straight line, while the handle may be thicker and more curved.',
        'A handle part of a knife 3D model might look like a cylindrical piece with a hole in the center for the blade to fit into.'],
    'lamp': [
        'There is no one-size-fits-all answer to this question, as the best way to segment the leg or wire part of a lamp in a depth map may vary depending on the.',
        'Since this is a depth map, you can segment the lampshade by finding the points in the depth map that correspond to the lampshade.'],
    'laptop': [
        'The keyboard feature of a laptop 3D model is that it is a separate object that can be moved around and positioned as desired.',
        'Laptop computer with screen open, viewed from above.'],
    'motorbike': ["The gas tank's motorbike would appear as a dark object in a grayscale depth map.",
                  'There is no easy answer for this question.',
                  'This sentence is describing a wheel on a motorcycle in a photograph.',
                  'There is no definitive answer to this question as it depends on the specific depth map and the desired outcome.',
                  'There is no definitive answer to this question since it will vary depending on the desired outcome.',
                  'The engine is the "heart" of the bike.'],
    'mug': [
        'This sentence is describing a depth map, which is a tool used in computer vision to create a representation of the surfaces of a scene from a set of digital images.',
        'Only the bottom part of this mug is recognized.'],
    'pistol': ['This is the part of the pistol depth map that shows the barrel.',
               'The part of the pistol that you would hold in your hand is the grip.',
               'Thesynonym of this sentence is: The trigger and guard of a gun.'],
    'rocket': ['ROCKET BODYThis is the body of a rocket.',
               'A fin is typically a thin, flat surface that is attached to the back end of a rocket.',
               'A nose cone on a rocket 3D model typically looks like a cone or pyramid shape.'],
    'skateboard': ['The depth map of the wheel on a skateboard is important.',
                   'Caption: The deck of a skateboard, viewed from the top.',
                   'This sentence is describing a strap or belt that goes around the foot of a skateboard.'],
    'table': ['A depth map of a table, showing the desktop at the top and the underside of the table at the bottom.',
              'The table is a rectangle with a light gray color.', 'The table is a rectangle with a light gray color.'],
}

best_vweight = {
    'airplane': [0.75, 0.75, 0.25, 0.25, 0.25, 0.50, 1.00, 0.25, 0.25, 0.25],
    'bag': [0.75, 0.75, 0.25, 0.75, 1.00, 0.25, 1.00, 0.50, 0.25, 0.25],
    'cap': [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    'car': [0.75, 0.75, 0.25, 0.25, 0.25, 0.75, 0.25, 0.75, 1.00, 0.25],
    'chair': [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    'earphone': [0.75, 0.75, 0.25, 0.25, 0.25, 0.25, 0.75, 0.50, 0.25, 0.50],
    'guitar': [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    'knife': [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    'lamp': [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    'laptop': [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    'motorbike': [0.75, 0.75, 0.25, 0.25, 0.50, 0.75, 0.25, 0.75, 1.00, 0.25],
    'mug': [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    'pistol': [0.75, 0.75, 0.25, 0.25, 0.25, 1.00, 0.25, 1.00, 0.75, 0.25],
    'rocket': [0.75, 0.75, 0.25, 0.25, 0.50, 1.00, 0.25, 0.50, 0.25, 0.75],
    'skateboard': [0.75, 0.75, 0.25, 0.50, 0.25, 1.00, 0.50, 0.25, 0.75, 1.00],
    'table': [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
}


# --------------------------------------------------------------------------- #
#  PartGeoZe v2  (partmodel/partgeozev2.py)                                    #
# --------------------------------------------------------------------------- #
# Everything the part-level v2 pipeline needs at evaluation time, searched once on the ShapeNetPart
# test features (the same protocol under which best_prompt / best_vweight above were found) and
# then frozen, so part_run.py stays free of magic numbers.  Every entry can be overridden from the
# command line (`python part_run.py --model v2 --th_f 0.6`); see probe_partition.py for the
# measurements behind the partition choice and the merge thresholds.
best_param_v2 = dict(
    part='kmeans',         # superpoints: kmeans | spectral | fps   (partmodel/spectral.py).
                           # MEASURED default.  Oracle IoU at 64 regions, mean over 7 categories:
                           #   fps 80.54 @ 0.65 ms | kmeans 82.86 @ 1.93 | kmeans+refine 83.00 @
                           #   2.19 | Nystrom-ortho 82.90 @ 10.57 | dense spectral 84.20 @ 37.31.
                           # The CUES carry the partition, not the clustering algorithm: the
                           # spectral machinery buys +1.2 oracle for 17x the time, and Nystrom
                           # needs 256 landmarks just to tie plain k-means.  'spectral' with
                           # --embed dense is still the quality ceiling if time does not matter.
    n_sp=32,               # superpoints per shape.  MEASURED on the end task, not on the
                           # partition: oracle IoU keeps climbing with resolution (69.86 at 16
                           # regions -> 87.23 at 128) while class-mIoU PEAKS AT 32 and then
                           # falls (16: 51.57 | 32: 54.18 | 64: 53.89 | 96: 53.64 | 128: 53.48).
                           # Fewer, larger regions average more features per region, and that
                           # denoising outweighs the lower ceiling.  Do not tune n_sp on oracle.
    knn=10,                # point neighbours: graph for the affinity, the merge and the boundary gate
    w_x=1.0, w_n=1.0, w_g=1.0,   # affinity cues: position, normal, FPFH (each on its own scale)
    w_v=1.0,               # concavity cue (LCCP): part seams are concave.  Needs oriented normals
    self_tune=True,        # Zelnik-Manor local scaling of the position cue
    n_ev=0,                # size of the spectral embedding (0 = the cluster count)
    embed='sparse',        # how the spectral embedding is solved, when part='spectral'.
                           # 'sparse'  subspace iteration on the kNN graph; the operator is only
                           #           APPLIED (a gather and a scatter), never formed.
                           # 'dense'   full eigh of an N x N matrix -- now strictly DOMINATED:
                           #           at matched region counts sparse equals or beats it at 2-6x
                           #           less time (68 regions: 83.69 @ 8.8 ms vs 84.20 @ 37.3;
                           #           103: 86.56 @ 19.5 vs 85.93 @ 37.2; 132: 87.78 vs 87.24).
                           # 'nystrom' landmark extension; needs 256 landmarks and 10.6 ms just
                           #           to tie plain k-means, so it is kept only for reference.
                           # 'lobpcg'  189 ms/shape -- slower than the full decomposition.
    sparse_iters=50,       # subspace-iteration steps for embed='sparse'.  Fewer steps leave
                           # the embedding less converged, which fragments clusters and so
                           # inflates the region count -- compare at matched REGIONS, never
                           # at matched n_sp (30 steps: 88.9 regions from n_sp=64).
    n_land=256,            # Nystrom landmarks
    ortho=False,           # Fowlkes' orthogonalised extension costs a second m x m eigh
                           # (10.69 vs 5.40 ms/shape) and k-means only needs the directions
    land='curve', seed='curve',   # Hilbert-curve stride instead of farthest-point sampling:
                           # FPS is a Python loop over the sample count, 2.89 -> 0.27 ms/shape
    refine=3,              # rounds of affinity-weighted boundary refinement.  This is the only
                           # route by which the CONCAVITY cue reaches a k-means partition, since
                           # k-means clusters per-point features and concavity is an edge
                           # quantity: boundary recall 71.12 -> 72.73 with it, 69.53 without any
                           # refinement.  More rounds do not help (6 rounds: 82.96).
    min_size=4,            # fragments smaller than this are absorbed by a neighbour
    split=True,            # cut every spectral cluster into its connected components
    center=True,           # merge on per-shape mean-centred features (see partgeozev2.py)
    th_f=0.5, th_n=0.3,    # merge admissibility: semantic (centred cosine) and geometric
    rounds=0,              # depth of the mutual-best-match hierarchy.  0 = MERGING OFF, and that
                           # is the measured default: on ShapeNetPart the merge costs -1.81
                           # class-mIoU (52.17 vs 53.98 for pooling over the same partition),
                           # because no cue separates adjacent same-part from different-part
                           # regions -- the best AUC over ten categories is 0.67 and the worst is
                           # 0.43.  Adjacent regions sit at cosine 0.99 either way.  See
                           # probe_partition.py.  Kept implemented and switchable (--rounds 10).
    alpha=1.0,             # recovery: 1 = every point takes its region feature outright
    gamma0=0.0,            # inter-region residual strength (0 = off, as in semseg)
)
