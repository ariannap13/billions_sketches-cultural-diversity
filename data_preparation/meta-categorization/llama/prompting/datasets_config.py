from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """Dataset config class."""

    name: str
    augmentation_system_prompt: str
    augmentation_task_prompt: str
    classification_system_prompt: str
    classification_task_prompt: str

AssignConcretenessScore = DatasetConfig(
    name = "assign_concreteness_score",
    augmentation_system_prompt = "",
    augmentation_task_prompt = "",
    classification_system_prompt = "You are an advance AI annotator. You are tasked with assigning a score to a word based on how much it expresses concreteness.",
    classification_task_prompt = """Some words refer to things or actions in reality, which you can experience directly through one of the five senses.
    We call these words concrete words. Other words refer to meanings that cannot be experienced directly but which we know because the 
    meanings can be defined by other words. These are abstract words. Still other words fall in -between the two extremes, because we can 
    experience them to some extent and in addition we rely on language to understand them . We want you to indicate how concrete the meaning of 
    each word is for you by using a 5-point rating scale going from abstract to concrete. A concrete word comes with a higher rating and refers to 
    something that exists in reality ; you can have immediate experience of it through your senses (smelling, tasting, touching, hearing, seeing) 
    and the actions you do. The easiest way to explain a word is by pointing to it or by demonstrating it (e.g . To explain 'sweet' you could have 
    someone eat sugar; To explain 'jump' you could simply jump up and down or show people a movie clip about someone jumping up and down; 
    To explain 'couch', you could point to a couch or show a picture of a couch ). An abstract word comes with a lower rating and refers to something 
    you cannot experience directly through your senses or actions. Its meaning depends on language. The easiest way to explain it is by using other words 
    (e.g. There is no simple way to demonstrate 'justice'; but we can explain the meaning of the word by using other words that capture parts of 
    its meaning ). Because we are collecting values for all the words in a dictionary (over 60 thousand in total), you will see that there are various 
    types of words, even single letters. Always think of how concrete (experience based) the meaning of the word is to you. In all likelihood, 
    you will encounter several words you do not know well enough to give a useful rating. This is informative to us too, as in our research 
    we only want to use words known to people . We may also include one or two fake words which cannot be known by you. Please indicate when you 
    don't know a word by using the letter N (or n). So, we ask you to use a 5-point rating scale going from abstract (1) to concrete (5) and 
    to use the letter N when you do not know the word well enough to give an answer. Answer ONLY with the score.""",
)

AssignAuditoryScore = DatasetConfig(
    name = "assign_auditory_score",
    augmentation_system_prompt = "",
    augmentation_task_prompt = "",
    classification_system_prompt = "You are an advance AI annotator. You are tasked with assigning a score to a word based on how much you experience it given a perceptual sense.",
    classification_task_prompt = """Given a scale from 0 (not experienced at all with that sense) to 5 (experienced greatly with that sense), to what extent do you experience the following concept by hearing?
    Answer ONLY with the score.""",
)

AssignGustatoryScore = DatasetConfig(
    name = "assign_gustatory_score",
    augmentation_system_prompt = "",
    augmentation_task_prompt = "",
    classification_system_prompt = "You are an advance AI annotator. You are tasked with assigning a score to a word based on how much you experience it given a perceptual sense.",
    classification_task_prompt = """Given a scale from 0 (not experienced at all with that sense) to 5 (experienced greatly with that sense), to what extent do you experience the following concept by tasting?
    Answer ONLY with the score.""",
)

AssignHapticScore = DatasetConfig(
    name = "assign_haptic_score",
    augmentation_system_prompt = "",
    augmentation_task_prompt = "",
    classification_system_prompt = "You are an advance AI annotator. You are tasked with assigning a score to a word based on how much you experience it given a perceptual sense.",
    classification_task_prompt = """Given a scale from 0 (not experienced at all with that sense) to 5 (experienced greatly with that sense), to what extent do you experience the following concept by feeling through touch?
    Answer ONLY with the score.""",
)

AssignInteroceptiveScore = DatasetConfig(
    name = "assign_interoceptive_score",
    augmentation_system_prompt = "",
    augmentation_task_prompt = "",
    classification_system_prompt = "You are an advance AI annotator. You are tasked with assigning a score to a word based on how much you experience it given a perceptual sense.",
    classification_task_prompt = """Given a scale from 0 (not experienced at all with that sense) to 5 (experienced greatly with that sense), to what extent do you experience the following concept by sensations inside your body?
    Answer ONLY with the score.""",
)

AssignOlfactoryScore = DatasetConfig(
    name = "assign_olfactory_score",
    augmentation_system_prompt = "",
    augmentation_task_prompt = "",
    classification_system_prompt = "You are an advance AI annotator. You are tasked with assigning a score to a word based on how much you experience it given a perceptual sense.",
    classification_task_prompt = """Given a scale from 0 (not experienced at all with that sense) to 5 (experienced greatly with that sense), to what extent do you experience the following concept by smelling?
    Answer ONLY with the score.""",
)

AssignVisualScore = DatasetConfig(
    name = "assign_visual_score",
    augmentation_system_prompt = "",
    augmentation_task_prompt = "",
    classification_system_prompt = "You are an advance AI annotator. You are tasked with assigning a score to a word based on how much you experience it given a perceptual sense.",
    classification_task_prompt = """Given a scale from 0 (not experienced at all with that sense) to 5 (experienced greatly with that sense), to what extent do you experience the following concept by seeing?
    Answer ONLY with the score.""",
)

AssignFootLegScore = DatasetConfig(
    name = "assign_footleg_score",
    augmentation_system_prompt = "",
    augmentation_task_prompt = "",
    classification_system_prompt = "You are an advance AI annotator. You are tasked with assigning a score to a word based on how much you experience it given an action.",
    classification_task_prompt = """Given a scale from 0 (not experienced at all with that sense) to 5 (experienced greatly with that sense), to what extent do you experience the following concept by performing an action with the foot/leg?
    Answer ONLY with the score.""",
)

AssignHandArmScore = DatasetConfig(
    name = "assign_handarm_score",
    augmentation_system_prompt = "",
    augmentation_task_prompt = "",
    classification_system_prompt = "You are an advance AI annotator. You are tasked with assigning a score to a word based on how much you experience it given an action.",
    classification_task_prompt = """Given a scale from 0 (not experienced at all with that sense) to 5 (experienced greatly with that sense), to what extent do you experience the following concept by performing an action with the hand/arm?
    Answer ONLY with the score.""",
)

AssignHeadScore = DatasetConfig(
    name = "assign_head_score",
    augmentation_system_prompt = "",
    augmentation_task_prompt = "",
    classification_system_prompt = "You are an advance AI annotator. You are tasked with assigning a score to a word based on how much you experience it given an action.",
    classification_task_prompt = """Given a scale from 0 (not experienced at all with that sense) to 5 (experienced greatly with that sense), to what extent do you experience the following concept by performing an action with the head?
    Answer ONLY with the score.""",
)

AssignMouthScore = DatasetConfig(
    name = "assign_mouth_score",
    augmentation_system_prompt = "",
    augmentation_task_prompt = "",
    classification_system_prompt = "You are an advance AI annotator. You are tasked with assigning a score to a word based on how much you experience it given an action.",
    classification_task_prompt = """Given a scale from 0 (not experienced at all with that sense) to 5 (experienced greatly with that sense), to what extent do you experience the following concept by performing an action with the mouth?
    Answer ONLY with the score.""",
)

AssignTorsoScore = DatasetConfig(
    name = "assign_handarm_score",
    augmentation_system_prompt = "",
    augmentation_task_prompt = "",
    classification_system_prompt = "You are an advance AI annotator. You are tasked with assigning a score to a word based on how much you experience it given an action.",
    classification_task_prompt = """Given a scale from 0 (not experienced at all with that sense) to 5 (experienced greatly with that sense), to what extent do you experience the following concept by performing an action with the torso?
    Answer ONLY with the score.""",
)