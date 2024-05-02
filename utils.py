def column_renamer(input):
    rename_dict = {
        "Unnamed: 0": "Run number",
        "angle_shfit_gandalf": "Angle between direction of shower and track",
        "distance_shfit_gandalf": "Distance between shower and track start",
        "dt_shfit_gandalf": "Time difference between shower and track start",
        "E.trks.len[:,0]": "Track length",
        "E.trks.dir.x[:,0]": "Track x-direction",
        "E.trks.dir.y[:,0]": "Track y-direction",
        "E.trks.dir.z[:,0]": "Track z-direction",
        "E.trks.pos.x[:,0]": "Track x-position",
        "E.trks.pos.y[:,0]": "Track y-position",
        "E.trks.pos.z[:,0]": "Track z-position",
        "E.trks.dir.x[:,1]": "Shower x-direction",
        "E.trks.dir.y[:,1]": "Shower y-direction",
        "E.trks.dir.z[:,1]": "Shower z-direction",
        "E.trks.pos.x[:,1]": "Shower x-position",
        "E.trks.pos.y[:,1]": "Shower y-position",
        "E.trks.pos.z[:,1]": "Shower z-position",
        "T.feat_Neutrino2020.cherCond_n_doms": "Number of detector spheres with unscattered light signals",
        "T.feat_Neutrino2020.gandalf_nHits": "Number of hits used in track reconstruction",
        "T.sum_mc_nu.by": "Inelasticity"
    }
    try:
        newname = rename_dict[input]
    except KeyError:
        newname = input
        
    return newname

def pdgid_converter(id):
    pdgid_dict = {
        12: "Electron neutrino",
        14: "Muon neutrino",
        16: "Tau neutrino",
        -12: "Anti electron neutrino",
        -14: "Anti muon neutrino",
        -16: "Anti tau neutrino"
    }
    return pdgid_dict[id]

def is_shower(row):
    id = row["pdgid"]
    is_cc = row["is_cc"]
    if((id == 14 or id == -14) and is_cc == 1.0):
        return False
    else:
        return True