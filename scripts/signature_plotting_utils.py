# Define the 6 base substitution types in their desired plotting order
MUTATION_TYPES = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']

# Define colors for each mutation type
MUTATION_TYPE_COLORS = {
    'C>A': '#01BFFD',  # Sky Blue
    'C>G': '#000000',  # Black
    'C>T': '#E62725',  # Red
    'T>A': '#CBC9C8',  # Silver/Grey
    'T>C': '#A0CE00',  # Lime Green
    'T>G': '#F298C3'   # Pink
}

# Define the 96 trinucleotide contexts in the standard plotting order
# Order: C>A, C>G, C>T, T>A, T>C, T>G
# Within each mutation type, order by flanking bases: A_A, A_C, A_G, A_T, C_A, ..., T_T
STANDARD_96_CONTEXTS = []
BASES = ['A', 'C', 'G', 'T']

# For C>X mutations (Ref is C)
ref_c_mutations = {
    'C>A': 'C>A',
    'C>G': 'C>G',
    'C>T': 'C>T'
}
# For T>X mutations (Ref is T)
ref_t_mutations = {
    'T>A': 'T>A',
    'T>C': 'T>C',
    'T>G': 'T>G'
}

# Iterate through the main mutation types in the specified order
for main_mutation_type_display in MUTATION_TYPES:
    # Determine the reference base and the mutation type string (e.g., "C>A")
    if main_mutation_type_display.startswith('C'):
        mutation_type_dict = ref_c_mutations
        current_ref = 'C'
    else: # Starts with 'T'
        mutation_type_dict = ref_t_mutations
        current_ref = 'T'

    # Find the specific mutation type string (e.g. C>A)
    # This logic is a bit redundant given MUTATION_TYPES already has the specific type
    # but it's okay for generation if we assume MUTATION_TYPES drives the order.
    
    # Correctly get the specific mutation type for context generation
    specific_mutation = main_mutation_type_display 

    for upstream_base in BASES:
        for downstream_base in BASES:
            context = f"{upstream_base}[{specific_mutation}]{downstream_base}"
            STANDARD_96_CONTEXTS.append(context)


def get_reordered_contexts(columns_from_csv):
    """
    Reorders a list of context names from a CSV according to the standard plotting order.

    Args:
        columns_from_csv (list): A list of context names (e.g., from DataFrame.columns).

    Returns:
        list: A new list of context names, reordered according to STANDARD_96_CONTEXTS.
              Contexts in STANDARD_96_CONTEXTS not found in columns_from_csv are omitted.
              Contexts in columns_from_csv not in STANDARD_96_CONTEXTS are currently ignored.
    """
    # Create a set for faster lookups from the input columns
    csv_contexts_set = set(columns_from_csv)
    
    reordered_list = [ctx for ctx in STANDARD_96_CONTEXTS if ctx in csv_contexts_set]
    
    # Optional: Check for contexts in columns_from_csv but not in STANDARD_96_CONTEXTS
    # extra_contexts = [ctx for ctx in columns_from_csv if ctx not in STANDARD_96_CONTEXTS_SET]
    # if extra_contexts:
    #     print(f"Warning: The following contexts from input are not part of the standard 96 contexts and will be ignored (or appended): {extra_contexts}")
        # reordered_list.extend(sorted(extra_contexts)) # Optionally append them

    return reordered_list

# Example to verify the length and first few/last few contexts
if __name__ == '__main__':
    print(f"Total generated contexts: {len(STANDARD_96_CONTEXTS)}")
    print("First 5 contexts:", STANDARD_96_CONTEXTS[:5])
    print("Contexts 15-19 (around C>A and C>G transition):", STANDARD_96_CONTEXTS[15:20])
    print("Last 5 contexts:", STANDARD_96_CONTEXTS[-5:])

    # Verify all are unique
    if len(STANDARD_96_CONTEXTS) == len(set(STANDARD_96_CONTEXTS)):
        print("All 96 contexts are unique.")
    else:
        print("Error: Duplicate contexts generated.")

    # Test get_reordered_contexts
    sample_csv_cols_sorted = sorted(STANDARD_96_CONTEXTS) # Lexicographically sorted
    sample_csv_cols_shuffled = STANDARD_96_CONTEXTS[:]
    import random
    random.shuffle(sample_csv_cols_shuffled)
    
    reordered_from_sorted = get_reordered_contexts(sample_csv_cols_sorted)
    reordered_from_shuffled = get_reordered_contexts(sample_csv_cols_shuffled)

    print("\nTesting get_reordered_contexts:")
    print(f"Reordering a sorted list matches original: {reordered_from_sorted == STANDARD_96_CONTEXTS}")
    print(f"Reordering a shuffled list matches original: {reordered_from_shuffled == STANDARD_96_CONTEXTS}")

    sample_csv_cols_subset = STANDARD_96_CONTEXTS[::2] # Take every other context
    reordered_subset = get_reordered_contexts(sample_csv_cols_subset)
    print(f"Reordering a subset matches original subset order: {reordered_subset == sample_csv_cols_subset}")

    sample_csv_cols_with_extra = STANDARD_96_CONTEXTS + ['X[Y>Z]A']
    reordered_with_extra = get_reordered_contexts(sample_csv_cols_with_extra)
    print(f"Reordering with extra context ignores it: {len(reordered_with_extra) == 96 and 'X[Y>Z]A' not in reordered_with_extra}")

    # Check the order of the first few generated contexts to match the prompt's example
    # Expected C>A: A[C>A]A, A[C>A]C, A[C>A]G, A[C>A]T, C[C>A]A...
    print("\nFirst 16 C>A contexts from generated list:")
    for i in range(16):
        print(STANDARD_96_CONTEXTS[i], end=', ' if (i+1)%4!=0 else '\n')
    
    # Expected C>G: A[C>G]A, A[C>G]C, A[C>G]G, A[C>G]T, C[C>G]A...
    print("\nFirst 16 C>G contexts from generated list (should be contexts 16-31):")
    for i in range(16, 32):
        print(STANDARD_96_CONTEXTS[i], end=', ' if (i-16+1)%4!=0 else '\n')

    # Check if the full list of 96 contexts is as expected
    # C>A contexts
    expected_contexts_prefix = [
        'A[C>A]A', 'A[C>A]C', 'A[C>A]G', 'A[C>A]T',
        'C[C>A]A', 'C[C>A]C', 'C[C>A]G', 'C[C>A]T',
        'G[C>A]A', 'G[C>A]C', 'G[C>A]G', 'G[C>A]T',
        'T[C>A]A', 'T[C>A]C', 'T[C>A]G', 'T[C>A]T',
    ]
    match = True
    for i in range(16):
        if STANDARD_96_CONTEXTS[i] != expected_contexts_prefix[i]:
            match = False
            break
    print(f"\nFirst 16 C>A contexts match example: {match}")