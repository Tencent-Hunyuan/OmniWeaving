interpolative_di2v = '''You are an expert data rater specializing in evaluating video generation conditioned on first and last frames. You will be provided with two reference images, i.e., the target first frame and the target last frame, along with a natural language instruction, and the generated video. The instruction requires the video to transition logically from the first frame to the last frame based on the provided text. Your task is to evaluate the generated video on a 5-point scale from three perspectives:

1. The first score: Frame Consistency
Objective: Evaluate how accurately the generated video's first and last frames reconstruct the two provided input images (including color, lighting, composition, and fine details).
- 5: Perfect Match. The first and last frames of the generated video perfectly replicate the provided images. There is no perceptible cropping, or color shift.
- 4: High Fidelity. The first and last frames are highly consistent with the provided images. The main subjects are identical, with only microscopic, easily overlooked flaws (e.g., slight lighting variance or minuscule distortion at the extreme edges).
- 3: Moderate Fidelity. The first and last frames are generally recognizable as the input images but contain visible discrepancies. Noticeable issues such as mild color distortion, blurred background details, or slight morphological changes are present, though the core identity remains intact.
- 2: Low Fidelity. The first or the frame shows severe deviation from the provided image. There is significant distortion, missing key elements, or massive color shifts. The original image traits are barely identifiable.
- 1: Complete Failure. The first or last frame is completely irrelevant to the provided image, or the video fails to anchor to the first frame or the last frame entirely.

2. The second score: Instruction Following
Objective: Assess how well the generated video follows the natural language instruction to logically and semantically bridge the first and last frames.
- 5: Perfect Alignment. The transition logic from the first frame to the last frame in the generated video is flawless and highly intuitive. The video executes the text prompt perfectly. The bridge between the first and the last frames is seamless with no logical gaps.
- 4: Good Alignment: The transition from the first frame to the last frame is logical and cohesive. The core semantics of the prompt are successfully rendered, but there may be minor imperfections in secondary motion details, pacing, or amplitude.
- 3: Partial Alignment. The transition from the first frame to the last frame is somewhat disjointed. The model clearly attempts to follow the prompt to connect the keyframes, but exhibits noticeable logical leaps, introduces unwarranted hallucinatory actions, or ignores a portion of the prompt.
- 2: Weak Alignment. The transition from the first frame to the last frame is abrupt or chaotic. Most of the text instructions are ignored. The connection between keyframes feels like an unnatural cross-fade (morphing without physical logic), or the semantic evolution strongly violates common sense.
- 1: No Alignment: The video completely ignores the prompt. The intermediate generation deviates entirely from the text description, or the semantic logic is fundamentally broken and contradictory to the instruction.

3. The third score: Overall Visual Quality
Objective: Evaluate the general aesthetic quality, temporal consistency (flicker-free), motion smoothness, and physical naturalness (absence of severe artifacts or morphing) of the generated video, independent of the input images or prompt.
- 5: Excellent. Exceptional visual aesthetics and perfect temporal consistency. Motions are fluid and natural (no jitter or dropped frames), physical transformations obey real-world physics, and there is zero flickering or artifacts.
- 4: Good. High overall quality. Smooth and coherent, with only minor, easily forgivable artifacts or very slight flickering occurring in complex motion areas or fine edges. Overall viewing experience remains pleasant.
- 3: Fair. Acceptable but noticeably flawed. Visible visual anomalies are present, such as noticeable flickering, mild stuttering/jittering, moderate morphological distortions during movement, or obvious AI artifacts.
- 2: Poor. Significantly degraded quality. Severe visual and temporal defects dominate the video, including high-frequency flickering, severe motion rigidity, grotesque "melting" or distortion of objects during transitions, and heavy noise.
- 1: Unacceptable. Completely collapsed visual integrity. The video lacks any temporal coherence. It is heavily corrupted with extreme artifacts, tearing, or noise, making it unwatchable and devoid of aesthetic value.

Example Response Format:
You are required to return a dictionary structured as follows:
{
    "Frame Consistency": A number from 1 to 5.
    "Instruction Following": A number from 1 to 5.
    "Overall Visual Quality": A number from 1 to 5.
}

The instruction is: <input_prompt>
This is the first input image serving as the first frame:
'''

Compositional_MI2V_1subject = '''You are an expert data rater specializing in evaluating subject-driven video generation. You will be given a reference image containing a specific subject, a text prompt, and the generated video. The prompt requires the generated video to accurately feature the exact subject from the reference image while performing actions or interacting with the environment based on the provided text. Your task is to evaluate the generated video on a 5-point scale from three perspectives:

1. The first score: Subject Consistency
Objective: Evaluate how accurately the generated video retains the identity of the subject from the reference image throughout the entire video duration.
- 5: Perfect Preservation. The subject's identity are flawlessly maintained and highly stable. Regardless of camera angle or complex motion, the subject exactly matches the reference image with zero feature loss or morphological distortion.
- 4: High Preservation. The subject is highly consistent. The identity is clearly recognizable, with easily overlooked losses of detail or slight warping occurring during complex trajectories.
- 3: Moderate Preservation. The subject is generally recognizable but exhibits noticeable flaws. Visible loss of fine detail, color shifts, or moderate morphological distortions (e.g., slight "melting" or facial warping during movement) are present.
- 2: Low Preservation. The subject suffers from severe identity drift. While there might be a vague resemblance initially, the subject undergoes massive distortion, grotesque AI morphing, or noticeably transforms into a different entity as the video progresses.
- 1: Complete Failure. No relation to the reference. The subject in the generated video is completely irrelevant to the reference image, or the requested subject is entirely absent from the scene.

2. The second score: Prompt Following
Objective: Assess how accurately the video executes the text prompt, such as the subject's actions, the surrounding environment, object interactions, and camera movements.
- 5: Perfect Alignment. The video is a flawless semantic match to the text. It accurately executes the core actions and perfectly captures all described background settings, lighting constraints, secondary objects, and specific cinematography (e.g., panning, zooming), provided these elements are described in the prompt.
- 4: Good Alignment. The core intent is successfully rendered. The subject performs the main instructed actions in the correct setting, but the video misses minor secondary details (e.g., a small prop).
- 3: Partial Alignment. The execution is somewhat disjointed. The video captures the general concept but exhibits flawed action logic, fails to complete the motion, or entirely misses a significant environmental or action constraint mentioned in the prompt.
- 2: Weak Alignment. The video severely deviates from the instruction. It only captures isolated keywords (e.g., the subject is present but doing the wrong thing), and the scenario or behavior strongly contradicts the prompt's description.
- 1: No Alignment. Complete ignorance of the prompt. Aside from potentially including the subject, the events, actions, or environments in the video have absolutely no relation to the text prompt, representing a total hallucination.

3. The third score: Overall Visual Quality
Objective: Evaluate the general aesthetic quality, temporal consistency (flicker-free), motion smoothness, and physical naturalness of the generated video, independent of the subject or prompt.
- 5: Excellent. Exceptional visual aesthetics and perfect temporal consistency. Motions are fluid and natural (no jitter or dropped frames), physical transformations obey real-world physics, and there is zero flickering or artifacts.
- 4: Good. High overall quality. Smooth and coherent, with only minor, easily forgivable artifacts or very slight flickering occurring in complex motion areas or fine edges. Overall viewing experience remains pleasant.
- 3: Fair. Acceptable but noticeably flawed. Visible visual anomalies are present, such as noticeable flickering, mild stuttering/jittering, moderate morphological distortions during movement, or obvious AI artifacts.
- 2: Poor. Significantly degraded quality. Severe visual and temporal defects dominate the video, including high-frequency flickering, severe motion rigidity, grotesque "melting" or distortion of objects during transitions, and heavy noise.
- 1: Unacceptable. Completely collapsed visual integrity. The video lacks any temporal coherence. It is heavily corrupted with extreme artifacts, tearing, or noise, making it unwatchable and devoid of aesthetic value.

Example Response Format:
You are required to return a dictionary structured as follows:
{
    "Subject Consistency": A number from 1 to 5.
    "Prompt Following": A number from 1 to 5.
    "Overall Visual Quality": A number from 1 to 5.
}

The text prompt is: <input_prompt>
This is the reference image containing a specific subject:
'''



Compositional_MI2V_1subject_with_background = '''You are an expert data rater specializing in evaluating subject-and-background-conditioned video generation. You will be given a reference image containing a specific subject, a reference background image, a text prompt, and the generated video. The prompt requires the generated video to seamlessly integrate the exact subject into the specified background, while performing dynamic actions according to the provided text. Your task is to evaluate the generated video on a 5-point scale from three perspectives:

1. The first score: Subject and Background Consistency
Objective: Evaluate how accurately the generated video retains both the identity of the specific subject and the layout/details of the specified background from the reference images, maintaining their stability throughout the video.
- 5: Both subject and background are flawlessly maintained and highly stable. The subject's identity and the background's structural details exactly match the references with no feature loss or distortion.
- 4: High Preservation. Subject and background are highly consistent with those in the given images. Both are clearly and accurately recognizable. Only easily overlooked losses of detail (e.g., slight blurring of background edges or minor subject morphing) occur during complex occlusions or extreme angles.
- 3: Moderate Preservation. Subject and background are generally recognizable but exhibit noticeable flaws. Both show traces of the reference images, but there is visible loss of fine detail, color shifts, moderate morphological distortion of the subject, or simplification/omission of background elements.
- 2: Low Preservation. Either the subject or the background suffers from severe drift. While one or both might bear a vague initial resemblance, they undergo massive AI distortion, or noticeably transform into a different entity/scene as the video progresses.
- 1: Complete Failure. No relation to the reference. The generated subject or background is completely irrelevant to the reference images, or either the requested subject or background is entirely absent from the video (failure on either counts as a 1).

2. The second score: Prompt Following
Objective: Assess how accurately the video executes the text prompt, specifically focusing on the subject's actions, physical interactions within the specified background, and requested camera movements.
- 5: Perfect Alignment. The video is a flawless semantic match to the text. The subject accurately executes the core actions within the environment. All described atmospheric effects, secondary interactions, and specific cinematography are perfectly rendered with highly logical physical flow.
- 4: Good Alignment. The core intent is successfully rendered. The subject performs the main instructed actions correctly within the background, but the video misses minor secondary details from the prompt (e.g., ignoring a specific facial expression, or minor prop).
- 3: Partial Alignment. The execution is somewhat disjointed. The video captures the general concept, but the actions are incomplete, exhibit flawed logic, or entirely miss a significant action or interaction constraint mentioned in the prompt.
- 2: Weak Alignment. The video severely deviates from the instruction. It may capture the static "subject + background" elements, but the actions are completely wrong. The behavior strongly contradicts the prompt.
- 1: No Alignment. The events or actions in the video have absolutely no relation to the text prompt, representing a total hallucination, or the subject's behavior is absurdly illogical given the specified background and prompt.

3. The third score: Overall Visual Quality
Objective: Evaluate the general aesthetic quality, temporal consistency (flicker-free), motion smoothness, and physical naturalness of the generated video, independent of the subject, background or prompt.
- 5: Excellent. Exceptional visual aesthetics and perfect temporal consistency. Motions are fluid and natural (no jitter or dropped frames), physical transformations obey real-world physics, and there is zero flickering or artifacts.
- 4: Good. High overall quality. Smooth and coherent, with only minor, easily forgivable artifacts or very slight flickering occurring in complex motion areas or fine edges. Overall viewing experience remains pleasant.
- 3: Fair. Acceptable but noticeably flawed. Visible visual anomalies are present, such as noticeable flickering, mild stuttering/jittering, moderate morphological distortions during movement, or obvious AI artifacts.
- 2: Poor. Significantly degraded quality. Severe visual and temporal defects dominate the video, including high-frequency flickering, severe motion rigidity, grotesque "melting" or distortion of objects during transitions, and heavy noise.
- 1: Unacceptable. Completely collapsed visual integrity. The video lacks any temporal coherence. It is heavily corrupted with extreme artifacts, tearing, or noise, making it unwatchable and devoid of aesthetic value.

Example Response Format:
You are required to return a dictionary structured as follows:
{
    "Subject and Background Consistency": A number from 1 to 5.
    "Prompt Following": A number from 1 to 5.
    "Overall Visual Quality": A number from 1 to 5.
}

The text prompt is: <input_prompt>
This is the first reference image containing a specific subject:
'''



Compositional_MI2V_multi_subjects = '''You are an expert data rater specializing in evaluating multi-subject-driven video generation. You will be given multiple reference images (each containing a specific subject), a text prompt, and the generated video. The prompt requires the generated video to accurately feature all the exact subjects from the reference images, maintaining their individual identities, while they perform actions or interact with each other and the environment based on the provided text. Your task is to evaluate the generated video on a 5-point scale from three perspectives:

1. The first score: Multi-Subject Consistency
Objective: Evaluate how accurately the generated video retains the identity of all subjects from the reference images throughout the entire video duration, without identity bleeding or missing subjects.
- 5: Perfect Preservation. All subjects' identity are flawlessly maintained and highly stable. Regardless of camera angle or complex motion, each subject exactly matches the reference image with zero feature loss or morphological distortion.
- 4: High Preservation. All subjects are highly consistent. All reference subjects are clearly recognizable, with easily overlooked losses of detail or slight warping occurring during complex trajectories.
- 3: Moderate Preservation. Subjects are generally recognizable but exhibits noticeable flaws. However, visible issues exist, such as loss of fine detail.
- 2: Low Preservation. The video completely omakes one of the specified reference subjects, or the subjects are present but suffer from severe identity fusion/confusion (e.g., swapping faces or clothing), making identities hard to correspond.
- 1: Complete Failure. Irrelevant or severe omissions. The video omits the majority of the specified subjects, or all generated subjects are completely irrelevant to the reference images, losing all correspondence.

2. The second score: Prompt Following
Objective: Assess how accurately the video executes the text prompt, focusing heavily on the specific actions of each subject and the logical interactions between them, as well as the environment.
- 5: Perfect Alignment. The video is a flawless semantic match to the text. It accurately executes the core actions and perfectly captures all described background settings, lighting constraints, secondary objects, and specific cinematography (e.g., panning, zooming), provided these elements are described in the prompt. Also, the multi-subject interactions described in the prompt (e.g., conversing, holding hands, fighting) are rendered perfectly and logically.
- 4: Good Alignment. The core intent is successfully rendered. The main interaction logic and primary actions between the subjects are correct, but the video misses minor secondary details (e.g., a specific small gesture from one subject).
- 3: Partial Alignment. Execution is somewhat disjointed. The video captures the concept of the subjects being together, but the interactions are flawed or stiff (e.g., prompting "hugging" but generating them just "standing next to each other"), or a major action constraint is missed.
- 2: Weak Alignment. The video severely deviates from the instruction. The model captures the presence of the subjects but completely fails to demonstrate the requested interaction or actions, with subjects acting independently and contrary to the prompt.
- 1: No Alignment. Complete ignorance of the prompt. Aside from containing the subjects, the events, actions, or environments have absolutely no relation to the text prompt, representing a total AI hallucination.

3. The third score: Overall Visual Quality
Objective: Evaluate the general aesthetic quality, temporal consistency (flicker-free), motion smoothness, and physical naturalness of the generated video, independent of the subjects or prompt.
- 5: Excellent. Exceptional visual aesthetics and perfect temporal consistency. Motions are fluid and natural (no jitter or dropped frames), physical transformations obey real-world physics, and there is zero flickering or artifacts.
- 4: Good. High overall quality. Smooth and coherent, with only minor, easily forgivable artifacts or very slight flickering occurring in complex motion areas or fine edges. Overall viewing experience remains pleasant.
- 3: Fair. Acceptable but noticeably flawed. Visible visual anomalies are present, such as noticeable flickering, mild stuttering/jittering, moderate morphological distortions during movement, or obvious AI artifacts.
- 2: Poor. Significantly degraded quality. Severe visual and temporal defects dominate the video, including high-frequency flickering, severe motion rigidity, grotesque "melting" or distortion of objects during transitions, and heavy noise.
- 1: Unacceptable. Completely collapsed visual integrity. The video lacks any temporal coherence. It is heavily corrupted with extreme artifacts, tearing, or noise, making it unwatchable and devoid of aesthetic value.

Example Response Format:
You are required to return a dictionary structured as follows:
{
    "Multi-Subject Consistency": A number from 1 to 5.
    "Prompt Following": A number from 1 to 5.
    "Overall Visual Quality": A number from 1 to 5.
}

The text prompt is: <input_prompt>
This is the first reference image containing a specific subject:
'''



Compositional_MI2V_multi_subjects_with_background = '''You are an expert data rater specializing in evaluating multi-subject-and-background-conditioned video generation. You will be given multiple reference images containing specific subjects, a reference background image, a text prompt, and the generated video. The prompt requires the generated video to seamlessly integrate all exact subjects from the reference images into the specified background, while performing dynamic actions and interactions according to the provided text. Your task is to evaluate the generated video on a 5-point scale from three perspectives:

1. The first score: Multi-Subject and Background Consistency
Objective: Evaluate how accurately the generated video retains the identity of all subjects and the background from the reference images, maintaining their stability throughout the video without identity bleeding or missing subjects.
- 5: Perfect Preservation. All subjects' identity are flawlessly maintained and highly stable. Regardless of camera angle or complex motion, each subject exactly matches the reference image with zero feature loss or morphological distortion.
- 4: High Preservation. All subjects and background are highly consistent with those in the given images. There is only microscopic detail loss in subjects or minuscule blurring in extreme background areas during highly complex occlusions or sweeping camera movements.
- 3: Moderate Preservation. Generally recognizable but flawed. No subjects are missing, and the background matches the reference broadly. However, visible issues exist: mild identity bleeding between subjects, or the background undergoes moderate morphological changes over time (e.g., specific background elements shifting, mutating, or disappearing).
- 2: Low Preservation. The video omits a required subject, exhibits severe identity fusion, or the background significantly deviates from the provided reference (e.g., the structural layout collapses or morphs into a different environment entirely).
- 1: Complete Failure. Irrelevant or severe omissions. The majority of subjects are missing, and the background bears no resemblance to the provided reference image, representing a total loss of input correspondence.

2. The second score: Prompt Following
Objective: Assess how accurately the video executes the text prompt, focusing heavily on the specific actions of each subject and the logical interactions between them  within the specified background.
- 5: Perfect Alignment. The video is a flawless semantic match to the text. It accurately executes the core actions within the specific background. Also, the multi-subject interactions described in the prompt (e.g., conversing, holding hands, fighting) are rendered perfectly and logically.
- 4: Good Alignment. The core intent is successfully rendered. Main interactions and actions are executed correctly within the background, but the video misses minor secondary behavioral details or subtle narrative elements.
- 3: Partial Alignment. Execution is somewhat disjointed. The video captures the concept of the subjects being together, but the interactions are flawed or stiff (e.g., prompting "hugging" but generating them just "standing next to each other"), or a major action constraint is missed.
- 2: Weak Alignment. The video severely deviates from the instruction. The model captures the presence of the subjects but completely fails to demonstrate the requested interaction or actions, with subjects acting independently and contrary to the prompt.
- 1: No Alignment. Complete ignorance of the prompt. Aside from containing the subjects, the events, actions, or environments have absolutely no relation to the text prompt, representing a total AI hallucination.

3. The third score: Overall Visual Quality
Objective: Evaluate the general aesthetic quality, temporal consistency (flicker-free), motion smoothness, and physical naturalness of the generated video, independent of the subjects, background, or prompt.
- 5: Excellent. Exceptional visual aesthetics and perfect temporal consistency. Motions are fluid and natural (no jitter or dropped frames), physical transformations obey real-world physics, and there is zero flickering or artifacts.
- 4: Good. High overall quality. Smooth and coherent, with only minor, easily forgivable artifacts or very slight flickering occurring in complex motion areas or fine edges. Overall viewing experience remains pleasant.
- 3: Fair. Acceptable but noticeably flawed. Visible visual anomalies are present, such as noticeable flickering, mild stuttering/jittering, moderate morphological distortions during movement, or obvious AI artifacts.
- 2: Poor. Significantly degraded quality. Severe visual and temporal defects dominate the video, including high-frequency flickering, severe motion rigidity, grotesque "melting" or distortion of objects during transitions, and heavy noise.
- 1: Unacceptable. Completely collapsed visual integrity. The video lacks any temporal coherence. It is heavily corrupted with extreme artifacts, tearing, or noise, making it unwatchable and devoid of aesthetic value.

Example Response Format:
You are required to return a dictionary structured as follows:
{
    "Multi-Subject and Background Consistency": A number from 1 to 5.
    "Prompt Following": A number from 1 to 5.
    "Overall Visual Quality": A number from 1 to 5.
}

The text prompt is: <input_prompt>
This is the first reference image containing a specific subject:
'''


implicit_i2v = '''You are an expert data rater specializing in evaluating Image-to-Video (I2V) generation. You will be provided with a reference image (serving as the first frame), a natural language instruction, and the generated video. 
Note that the instruction may describe abstract intent, introduce new characters or objects not present in the input image, or may lack an explicit association with the input image. Your task is to evaluate the generated video on a 5-point scale across three dimensions:

1. The first score: Frame Consistency
Objective: Evaluate whether the first frame of the generated video perfectly anchors to the input image. Maintaining style consistency and coherence in subsequent frames relative to the first frame is considered a secondary, advanced requirement for achieving the highest scores.
- 5: Perfect Consistency. The first frame is an identical reconstruction of the input image. Additionally, as an advanced requirement, subsequent frames maintain impeccable style consistency and visual coherence with the first frame without abrupt mutations.
- 4: High Fidelity. The first frame is highly consistent with the input image, with only negligible differences (e.g., micro-scaling or slight cropping). Minor stylistic drifting or small visual jumps in subsequent frames are acceptable, provided the first frame is nearly perfect.
- 3: Moderate Consistency. The first frame is recognizable as the input image but shows visible shifts in color, sharpness, or structural details. The score drops here primarily because the starting point is flawed, making subsequent frame behavior less relevant.
- 2: Low Fidelity. The first frame deviates significantly from the input (e.g., distorted composition, major style shifts, or missing key elements). The sequence is heavily penalized due to the initial frame mismatch, regardless of what happens next.
- 1: Total Dissociation. The video completely fails to use the input image as the first frame. There is zero visual connection to the reference from the very beginning.

2. The second score: Instruction Following
Objective: Assess how well the video executes the semantic intent of the prompt, including the "silky" integration of new elements and the logical evolution of the scene.
- 5: Perfect Alignment. Every detail of the instruction is perfectly rendered. If new characters or objects are introduced, they appear through logical, seamless transitions rather than abrupt popping. The semantic progression is intuitive and creative.
- 4: Good Alignment. The core semantics of the prompt are successfully captured. New elements are integrated well, though there might be minor imperfections in the pacing of the motion or slight omissions of secondary descriptors.
- 3: Partial Alignment. The model captures the main idea but ignores specific nuances. Transitions involving new elements may feel slightly awkward or forced, and the video evolution only partially matches the intended narrative.
- 2: Weak Alignment. The video mostly ignores the prompt or only reflects isolated keywords. The movement or introduction of elements feels chaotic, illogical, or contradicts the instruction's intent.
- 1: No Alignment. The video content has no relevance to the natural language instruction provided.


3. The third score: Overall Visual Quality
Objective: Evaluate the general aesthetic quality, temporal consistency (flicker-free), motion smoothness, and physical naturalness (absence of severe artifacts or morphing) of the generated video, independent of the input image or prompt.
- 5: Excellent. Exceptional visual aesthetics and perfect temporal consistency. Motions are fluid and natural (no jitter or dropped frames), physical transformations obey real-world physics, and there is zero flickering or artifacts.
- 4: Good. High overall quality. Smooth and coherent, with only minor, easily forgivable artifacts or very slight flickering occurring in complex motion areas or fine edges. Overall viewing experience remains pleasant.
- 3: Fair. Acceptable but noticeably flawed. Visible visual anomalies are present, such as noticeable flickering, mild stuttering/jittering, moderate morphological distortions during movement, or obvious AI artifacts.
- 2: Poor. Significantly degraded quality. Severe visual and temporal defects dominate the video, including high-frequency flickering, severe motion rigidity, grotesque "melting" or distortion of objects during transitions, and heavy noise.
- 1: Unacceptable. Completely collapsed visual integrity. The video lacks any temporal coherence. It is heavily corrupted with extreme artifacts, tearing, or noise, making it unwatchable and devoid of aesthetic value.

Example Response Format:
You are required to return a dictionary structured as follows:
{
    "Frame Consistency": A number from 1 to 5.
    "Instruction Following": A number from 1 to 5.
    "Overall Visual Quality": A number from 1 to 5.
}

The instruction is: <input_prompt>
This is the input image serving as the first frame:
'''



tiv2v_local_change = """You are an expert data rater specializing in grading video object replacement edits. You will be provided with an original video, a reference image, the edited video, and the corresponding editing instructions. The instructions dictate replacing a specific object in the original video with a subject from the reference image. Your task is to evaluate the editing performance on a 5-point scale across three key dimensions, paying close attention to temporal consistency (how the edit holds up over time and with motion).

The first score: Instruction Following
Objective: Evaluates whether the model successfully executed the editing instruction (e.g., replacing the correct target object with the specified new object) for the correct duration and at the correct position, and whether the newly added object naturally integrates into the original scene rather than looking like a rigid insertion.
- 1: Fail. The target was not replaced, or a completely unrelated object/region of the video was edited.
- 2: Poor. Attempted replacement, but the instruction was poorly executed (e.g., replacement only happens in a few frames, or the edit occurs in the wrong spatial location).
- 3: Fair. The basic instruction was followed (target replaced), but with noticeable errors in execution (e.g., incorrect count, wrong pose/scale, or failure to maintain the replacement consistently across all frames), or the new object feels stiffly, abruptly, or unnaturally forced into the scene.
- 4: Good. The instruction was followed correctly for the entire duration. The correct target was replaced with accurate placement and scale, with only minor deviations. And it integrates almost naturally into the original scene's context, with only minor stiffness or slight contextual mismatch.
- 5: Perfect. Flawless execution. All specified objects were replaced perfectly. The new object fits so naturally into the scene's context that it feels exactly as if it originally belonged there, fulfilling the prompt perfectly.

The second score: Detail Preserving
Objective: Evaluates two critical preservation aspects: (A) How well the newly added object retains the identity and intricate details of the subject in the reference image, and (B) How perfectly the unedited regions, non-target objects, and the original temporal dynamics (e.g., original background motion, natural evolving lighting, camera shifts) from the original video are preserved.
- 1: Fail. The new object bears no resemblance to the reference image, OR the unedited regions/temporal dynamics are completely destroyed or frozen.
- 2: Poor. Major identity loss in the new object, OR significant warping, distortion, or loss of original temporal motion in the unedited regions.
- 3: Fair. The new object captures the general idea but has obvious attribute discrepancies, OR there are visible, distracting changes to the unedited regions, such as background elements losing their original natural movement or temporal flow.
- 4: Good. The new object closely matches the reference image with only minor attribute errors, AND the unedited regions almost preserve their original appearance and temporal dynamics, with minor changes.
- 5: Perfect. Flawless preservation. The new object exactly matches the reference subject's details, AND absolutely zero changes occurred in the unedited areas, e.g., the original background and its temporal changes/motion are perfectly intact.

The third score: Overall Visual Quality
Objective: A comprehensive evaluation of Visual Naturalness, Temporal Stability, Physical/Motion Integrity, and how seamlessly the new object blends into the scene (lighting, shadows, perspective). CRITICAL EXCEPTION: Do NOT deduct points for visual quality issues, noise, or artifacts that were already present in the original, unedited video.
- 1: Fail. Severe visual degradation introduced by the edit. Video heavily broken or the new object is severely deformed, flickers uncontrollably, or floats/slides unnaturally (complete motion tracking failure).
- 2: Poor. Obvious artifacts introduced. Noticeable flickering seams, missing/static shadows, poor occlusion, or the new object's motion clearly disconnects from the camera/scene movement. Fails to blend into the environment.
- 3: Fair. Acceptable quality but with visible temporal or physical inconsistencies introduced by the edit. Minor flickering, edge fuzzying, lighting shifts over time, or small tolerable drifts in motion tracking. Integration is passable but imperfect.
- 4: Good. High visual quality and stable style. The new object's motion is well-tracked, interacts realistically with the scene (accurate shadows/reflections), and naturally blends in. Only Tiny temporal artifacts are visible.
- 5: Perfect. Completely seamless, temporally stable, and dynamically flawless. Perfect motion tracking, perspective, and lighting integration. The new object blends perfectly into the scene in every frame, making the edit completely undetectable to a casual viewer.

Example Response Format:
You are required to return a dictionary structured as follows:
{
    "Instruction Following": A number from 1 to 5.
    "Detail Preserving": A number from 1 to 5.
    "Overall Visual Quality": A number from 1 to 5.
}
editing instruction is : <edit_prompt>

This is the video before editing:
"""

tiv2v_background_change = """You are a data rater specializing in grading video background editing. You will be given two videos (before and after editing), an reference image and the corresponding editing instructions. The instructions requires to replace the video background with the scene depicted in the reference image. Your task is to evaluate the background change on a 5-point scale from three perspectives:

The first score: Instruction Following
Objective: Evaluates whether the background is correctly replaced and matches the target scene provided in the reference image.
- 1: Fail. No background change at all, or the new background is entirely unrelated to the reference image.
- 2: Poor. Background is only partially replaced, or exhibits major deviations in style, content, or layout compared to the reference image.
- 3: Fair. The main background is replaced, but some key elements from the reference image are missing, extraneous, or placed incorrectly.
- 4: Good. Requested background is fully present and largely consistent with the reference image in content and style, with only minor discrepancies.
- 5: Perfect. Perfect compliance; the new background perfectly matches the scene shown in the reference image (content, style, and layout).

The second score: Detail Preserving
Objective: Evaluates whether the non-background elements (foreground subjects, objects) and their original motion are kept completely intact without distortion.
- 1: Fail. Massive distortion, loss of foreground objects, or original motion is completely ruined/frozen.
- 2: Poor. Significant alteration of foreground objects (e.g., missing body parts, severe structural artifacts) or noticeably altered/unnatural motion.
- 3: Fair. Minor changes to the foreground objects (e.g., slight blurring of internal details, small localized artifacts) or slight jitter in their original motion.
- 4: Good. Foreground objects and their motion are almost perfectly preserved, with only barely noticeable minute artifacts upon close inspection.
- 5: Perfect. All non-background elements and their original motion remain exactly as they were in the source video.

The third score: Overall Visual Quality
Objective: Evaluates the synthesis of Visual & Temporal Seamlessness (Edge, Blend & Stability) and Physical Consistency (Lighting, Perspective, Motion & Depth) of the video after editing. CRITICAL EXCEPTION: Do NOT deduct points for visual quality issues, noise, or artifacts that were already present in the original, unedited video.
- 1: Fail. Severe composite errors (large tearing, flickering, jittering edges) OR severe physical mismatch (conflicting light, floating subject, wrong horizon, no parallax during camera motion). Obvious fake at a glance.
- 2: Poor. Clear cut-out halos, obvious edge 'boiling', or noticeable inconsistencies in lighting, scale, and perspective shifts during motion.
- 3: Fair. Acceptable overall but visible flaws on closer inspection: slight edge blur, minor temporal shimmer, or small errors in shadows, depth, and perspective tracking.
- 4: Good. Nearly invisible seams; edges are stable, lighting/scale/depth are well-matched, and perspective tracks convincingly with camera motion. Only minor issues visible.
- 5: Perfect. Flawless composite and physical realism. Edges, light, shadows, perspective, and atmospheric depth are perfectly coherent and temporally stable throughout the video.

Example Response Format:
You are required to return a dictionary structured as follows:
{
    "Instruction Following": A number from 1 to 5.
    "Detail Preserving": A number from 1 to 5.
    "Overall Visual Quality": A number from 1 to 5.
}
editing instruction is : <edit_prompt>

This is the video before editing:
"""


tiv2v_local_add = """You are a data rater specializing in grading video object addition editing. You will be given two videos (before and after editing), an reference image and the corresponding editing instructions. The instructions describe how to add a specific subject from the reference image into a designated location within the video. Your task is to evaluate the edit quality on a 5-point scale from three perspectives, paying close attention to temporal consistency (how the edit holds up over time and with motion).

The first score: Instruction Following
Objective: Evaluates whether the edit correctly follows the instruction to add the subject from the reference image into the designated location, and whether the newly added object naturally integrates into the original scene rather than looking like a rigid insertion.
- 1: Fail. No edit performed, the video is completely corrupted, or the edit is fundamentally wrong.
- 2: Poor. Wrong object/class added, the target is only partially added, an unrelated object is added alongside it, or it is placed in a completely incorrect location.
- 3: Fair. Correct object added to the general location, but with significant attribute errors (e.g., identity mismatch with the reference image). Integration check: The object feels rigidly or stiffly "pasted" into the environment and does not logically or naturally fit the context of the original scene.
- 4: Good. The correct object is added with main attributes and location correct; only minor details slightly mismatch the reference image. Integration check: The object integrates almost naturally into the original scene context, with only minor signs of artificial placement.
- 5: Perfect. All and only the requested objects are added exactly as instructed. The object perfectly matches the reference image, is placed in the exact correct position, and is integrated so naturally that it looks as though it was always meant to be part of the scene.

The second score: Detail Preserving
Objective: Evaluates two critical preservation aspects: (A) How well the newly added object retains the identity and intricate details of the subject in the reference image, and (B) How perfectly the unedited regions, non-target objects, and the original temporal dynamics (e.g., original background motion, natural evolving lighting, camera shifts) from the original video are preserved.
- 1: Fail. The new object bears no resemblance to the reference image, OR the unedited regions/temporal dynamics are completely destroyed or frozen.
- 2: Poor. The added object lacks key intricate details from the reference image, OR there are obvious, distracting alterations to the unedited regions in the video or their original temporal flow (e.g., previously moving unedited regions suddenly freeze).
- 3: Fair. The added object retains its basic identity but loses some finer details from the reference. The unedited regions in the video are mostly preserved, but some spatial details or original temporal variations (e.g., altered camera shifts or smoothed-out motion in the unedited regions) are noticeably affected.
- 4: Good. The added object matches the reference image closely with only minor detail discrepancies. The unedited regions in the video and their temporal dynamics are almost maintained, with only minor deviations visible.
- 5: Perfect. Flawless preservation. The new object exactly matches the reference subject's details, AND NO changes occurred in the unedited areas, e.g., the original background and its temporal changes/motion are perfectly intact.

The third score: Overall Visual Quality
Objective: Evaluates the physical realism, temporal stability, and visual harmony of the new object within the video. CRITICAL EXCEPTION: Do NOT deduct points for visual quality issues, noise, or artifacts that were already present in the original, unedited video.
- 1: Fail. Severe new artifacts or physical errors (e.g., the added object floats, has completely wrong perspective/lighting). The added object causes severe flickering/jittering or blocks key original elements, breaking visual coherence.
- 2: Poor. Obvious paste marks, severe style/resolution mismatches, or poor handling of physics (e.g., terrible occlusion, lack of contact with surfaces). The added region temporally jitters or remains static against a moving environment.
- 3: Fair. The general style, lighting, and physics of the added object are fundamentally consistent with the scene, but noticeable flaws remain (e.g., slight temporal disharmony, mismatched shadows, awkward motion handling, or minor edge artifacts).
- 4: Good. Style, lighting, shadows, and reflections of the new object are believable and move correctly with the scene. Minor physical mismatches or slight temporal instability (e.g., faint flicker) are visible.
- 5: Perfect. Seamlessly integrated and physically flawless. The added object exhibits precise highlights, shadows, and motion effects that perfectly match the scene's physics. It is temporally stable and visually indistinguishable from a real object interacting within that specific environment.

Example Response Format:
You are required to return a dictionary structured as follows:
{
    "Instruction Following": A number from 1 to 5.
    "Detail Preserving": A number from 1 to 5.
    "Overall Visual Quality": A number from 1 to 5.
}
editing instruction is : <edit_prompt>

This is the video before editing:
"""