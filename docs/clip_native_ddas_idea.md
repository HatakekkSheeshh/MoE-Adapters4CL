# CLIP-Native DDAS Replacement for MoE-Adapters4CL

## 1. Short Idea

Replace the current DDAS selector, which uses AlexNet features plus task-specific AutoEncoders, with a CLIP-native task/domain selector. The new selector uses CLIP image embeddings, and optionally CLIP text embeddings, to decide whether an input belongs to a previously learned task/domain or should fall back to the original CLIP zero-shot path.

## 2. Current Limitation

In the current `mtil` pipeline, DDAS works as follows:

```text
input image
 -> AlexNet feature extractor
 -> task-specific AutoEncoders
 -> reconstruction loss per task
 -> select task with lowest loss
 -> use CLIP + MoE-Adapter for that task
```

This creates a mismatch:

```text
main model representation = CLIP
task/domain selector representation = AlexNet
```

AlexNet is older and is not aligned with CLIP's vision-language embedding space. As a result, DDAS may select the wrong task/domain even when CLIP itself has a more meaningful representation of the input.

## 3. Proposed Method

Use CLIP image embeddings to build a prototype for each learned task/domain.

During or after training each task:

```text
prototype_task = mean(CLIP_image_embedding(training_images_of_task))
```

At inference:

```text
input image
 -> CLIP image encoder
 -> image embedding
 -> compare with all task prototypes
 -> select nearest task if confidence is high
 -> otherwise fall back to original CLIP zero-shot
```

The selector can use cosine similarity:

```text
score(task_i) = cosine(image_embedding, prototype_task_i)
```

If the best score is above a threshold, use the corresponding MoE-Adapter task route. If the best score is below the threshold, treat the input as out-of-distribution and use the original CLIP route.

## 4. Optional Text-Aware Variant

Because CLIP has both image and text encoders, the selector can also use text-side task prototypes.

For each task, build a text prototype from class names:

```text
text_prototype_task = mean(CLIP_text_embedding(prompt(class_name)))
```

Then combine image-domain and text-domain evidence:

```text
final_score(task_i) =
    alpha * cosine(image_embedding, image_prototype_task_i)
  + (1 - alpha) * cosine(image_embedding, text_prototype_task_i)
```

This variant tests whether class-name semantics help task/domain selection.

## 5. Implementation Plan

### Step 1: Add prototype storage

Create a module such as:

```text
mtil/src/models/clip_native_selector.py
```

It should support:

```text
build_image_prototype(model, dataloader, task_id)
save_prototypes(path)
load_prototypes(path)
select_task(image_embedding, prototypes, threshold)
```

### Step 2: Build prototypes after each task

After training a task in `mtil/src/models/finetune.py`, collect CLIP image embeddings for a small subset or full training set and save the mean vector.

### Step 3: Replace AutoEncoder selection during eval

Modify the eval path in:

```text
mtil/src/models/evaluation.py
```

Current path:

```text
AlexNet feature -> AutoEncoder loss -> best_router
```

New path:

```text
CLIP image embedding -> prototype similarity -> task_id
```

### Step 4: Keep fallback behavior

Preserve the existing DDAS idea:

```text
known task/domain -> use MoE-Adapter
unknown/OOD input -> use original CLIP
```

The difference is only how the task/domain is selected.

## 6. Experimental Design

### Main Baselines

Compare:

| Method | Selector |
| --- | --- |
| Original MoE-Adapters4CL | AlexNet + AutoEncoder DDAS |
| CLIP-native image prototype | CLIP image embedding + task prototypes |
| CLIP-native image-text prototype | CLIP image + text prototype combination |
| No DDAS fallback | Always use selected adapter |
| CLIP zero-shot | No adapter |

### Metrics

Report:

```text
average accuracy
forgetting
task/domain routing accuracy
OOD fallback accuracy
number of selector parameters
inference overhead
```

### Ablations

Important ablations:

```text
prototype built from full train set vs few samples
cosine threshold values
image-only vs image-text selector
frozen CLIP embedding vs adapter-aware embedding
per-dataset threshold vs global threshold
```

## 7. Expected Contribution

The contribution is not a new CLIP backbone. The contribution is a cleaner task/domain selection mechanism for continual vision-language learning:

```text
CLIP-native selector instead of external AlexNet AutoEncoder selector
lower selector complexity
better alignment with CLIP representation
potentially better routing and OOD fallback
```

## 8. Paper Framing

Possible paper title:

```text
CLIP-Native Distribution Selection for Continual Vision-Language Learning
```

Core claim:

```text
Task/domain selection in continual CLIP models should be performed in the same vision-language embedding space as the backbone model. Replacing external AutoEncoder-based DDAS with CLIP-native prototype selection improves routing reliability while reducing selector complexity.
```

## 9. Risks

Potential weaknesses:

```text
CLIP prototypes may be too coarse for visually similar domains.
Global threshold tuning may not generalize across datasets.
If adapter-modified CLIP embeddings drift, stored prototypes may become stale.
Few-shot prototypes may be noisy.
```

Mitigations:

```text
use normalized embeddings
maintain running prototypes after each task
evaluate per-task and global thresholds
add confidence calibration
compare image-only and image-text selectors
```

## 10. Why This Is a Reasonable Extension

The original method already shows that selecting between MoE-Adapters and original CLIP is important. This idea keeps that high-level design but replaces the selector with a representation that is native to CLIP. It is easier to explain, removes the AlexNet dependency, and directly tests whether CLIP's own embedding space is better for continual task/domain routing.
