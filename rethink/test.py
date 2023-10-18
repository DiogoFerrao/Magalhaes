import math

# def adjust_augmentation_probability(p, class_counts):
#     # Calculate class weights (inverse of class percentages)
#     total_count = sum(class_counts.values())
#     class_weights = {cls: total_count / count for cls, count in class_counts.items()}
    
#     # Calculate adjusted augmentation rates
#     adjusted_rates = {cls: p * weight / sum(class_weights.values()) for cls, weight in class_weights.items()}

#     # Ensure that the weighted sum of the individual class probabilities equals p
#     scaled_rates = {}
#     for cls in class_counts:
#         scaled_rates[cls] = adjusted_rates[cls] * p * total_count / sum([class_counts[c] * adjusted_rates[c] for c in class_counts])

#     return scaled_rates



# def adjust_augmentation_probability(p, class_counts):
#     # Calculate class weights (inverse of class percentages)
#     total_count = sum(class_counts.values())
#     class_weights = {cls: total_count / count for cls, count in class_counts.items()}
#     print("class_weights", class_weights)
    
#     # # Calculate adjusted augmentation rates
#     # adjusted_rates = {cls: p * weight / sum(class_weights.values()) for cls, weight in class_weights.items()}

#     # print("adjusted_rates", adjusted_rates)

#     # # Calculate the expected number of samples to be augmented
#     # expected_augmented = sum([count * rate for cls, count in class_counts.items() for cls_rate, rate in adjusted_rates.items() if cls == cls_rate])

#     # print("expected_augmented", expected_augmented)

#     # # Calculate the difference between expected and desired augmented samples
#     # difference = p * total_count - expected_augmented

#     expected_augmented = total_count * p

#     # Calculate a value "x" for which expected_augmented = x (class_weights[cls] * class_counts[cls])    

#     # Ensure that the weighted sum of the individual class probabilities equals p
#     scaled_rates = {}
#     for cls in class_counts:
#         scaled_rates[cls] = adjusted_rates[cls] * p * total_count / sum([class_counts[c] * adjusted_rates[c] for c in class_counts])

#     print("scaled_rates", scaled_rates)

#     result = {}
#     for cls, rate in scaled_rates.items():
#         if rate > 1:
#             augs_per_signal = math.ceil(rate)
#             residual_p = rate / augs_per_signal
#         else:
#             augs_per_signal = 1
#             residual_p = rate
        
#         result[cls] = {
#             "augs_per_signal": augs_per_signal,
#             "p": residual_p
#         }

#     return result

# def adjust_augmentation_probability(p, class_counts):
#     total_count = sum(class_counts.values())
    
#     # Calculate the desired number of augmented samples for each class
#     desired_augmented_counts = {cls: p * total_count for cls in class_counts.keys()}
#     print("desired_augmented_counts", desired_augmented_counts)
    
#     # Calculate how many augmented samples we need for each class
#     needed_augmentations = {cls: desired - count for cls, count, desired in zip(class_counts.keys(), class_counts.values(), desired_augmented_counts.values())}
    
#     result = {}
#     for cls, needed in needed_augmentations.items():
#         if needed <= 0:
#             result[cls] = {"augs_per_signal": 1, "p": 0.0}
#             continue
        
#         # Determine how many times each sample should be augmented
#         augs_per_signal = math.ceil(needed / class_counts[cls])
        
#         # Calculate the probability for the last round of augmentations
#         residual_p = (needed / class_counts[cls]) % 1
        
#         # If we only need one round of augmentations, use the calculated probability
#         if augs_per_signal == 1:
#             residual_p = needed / class_counts[cls]
        
#         result[cls] = {
#             "augs_per_signal": augs_per_signal,
#             "p": residual_p
#         }

#     return result

def adjust_augmentation_probability(total_new_entries, class_counts, total_count, aggressiveness=1.0):
    # Calculate class weights (inverse of class percentages)
    # total_count = sum(class_counts.values())
    class_weights = {cls: (total_count / count)**aggressiveness for cls, count in class_counts.items()}
    
    total_weight = sum(class_weights.values())
    class_percentages = {cls: (weight / total_weight) * 100 for cls, weight in class_weights.items()}


    new_entries_per_class = {cls: round(total_new_entries * (percentage / 100)) for cls, percentage in class_percentages.items()}

    sum_new_entries = sum(new_entries_per_class.values())
    print("Sum: ", sum_new_entries)

    if sum_new_entries > total_new_entries:
        # Remove the difference from the class with the biggest percentage
        biggest_percentage = max(class_percentages.values())
        biggest_class = [cls for cls, percentage in class_percentages.items() if percentage == biggest_percentage][0]
        new_entries_per_class[biggest_class] -= sum_new_entries - total_new_entries
    elif sum_new_entries < total_new_entries:
        # Add the difference to the class with the smallest percentage
        smallest_percentage = min(class_percentages.values())
        smallest_class = [cls for cls, percentage in class_percentages.items() if percentage == smallest_percentage][0]
        new_entries_per_class[smallest_class] += total_new_entries - sum_new_entries

    sum_new_entries = sum(new_entries_per_class.values())
    print("Sum: ", sum_new_entries)


    return new_entries_per_class



class_counts = {'person': 5901.0, 'bicycle': 383.0, 'car': 5187.0, 'motorcycle': 609.0, 'siren': 218.0, 'bus': 279.0, 'truck': 286.0}
p = 0.4
print(adjust_augmentation_probability(10000, class_counts, 11384, 2))

