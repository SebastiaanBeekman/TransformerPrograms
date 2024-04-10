import numpy as np
import pandas as pd


def select_closest(keys, queries, predicate):
    scores = [[False for _ in keys] for _ in queries]
    for i, q in enumerate(queries):
        matches = [j for j, k in enumerate(keys) if predicate(q, k)]
        if not (any(matches)):
            scores[i][0] = True
        else:
            j = min(matches, key=lambda j: len(matches) if j == i else abs(i - j))
            scores[i][j] = True
    return scores


def select(keys, queries, predicate):
    return [[predicate(q, k) for k in keys] for q in queries]


def aggregate(attention, values):
    return [[v for a, v in zip(attn, values) if a][0] for attn in attention]


def aggregate_sum(attention, values):
    return [sum([v for a, v in zip(attn, values) if a]) for attn in attention]


def run(tokens):

    # classifier weights ##########################################
    classifier_weights = pd.read_csv(
        "output/length/rasp/reverse/trainlength40/s0/reverse_weights.csv",
        index_col=[0, 1],
        dtype={"feature": str},
    )
    # inputs #####################################################
    token_scores = classifier_weights.loc[[("tokens", str(v)) for v in tokens]]

    positions = list(range(len(tokens)))
    position_scores = classifier_weights.loc[[("positions", str(v)) for v in positions]]

    ones = [1 for _ in range(len(tokens))]
    one_scores = classifier_weights.loc[[("ones", "_") for v in ones]].mul(ones, axis=0)

    # attn_0_0 ####################################################
    def predicate_0_0(q_position, k_position):
        if q_position in {0, 39}:
            return k_position == 11
        elif q_position in {1, 42}:
            return k_position == 27
        elif q_position in {2, 26}:
            return k_position == 9
        elif q_position in {3, 45}:
            return k_position == 36
        elif q_position in {11, 4, 6}:
            return k_position == 25
        elif q_position in {5}:
            return k_position == 13
        elif q_position in {7}:
            return k_position == 26
        elif q_position in {8}:
            return k_position == 18
        elif q_position in {9}:
            return k_position == 21
        elif q_position in {10}:
            return k_position == 29
        elif q_position in {12, 13}:
            return k_position == 22
        elif q_position in {14, 17, 20, 22, 27, 29}:
            return k_position == 6
        elif q_position in {33, 46, 15, 19, 21, 23, 24}:
            return k_position == 5
        elif q_position in {16, 37, 38}:
            return k_position == 1
        elif q_position in {18, 35, 36, 31}:
            return k_position == 3
        elif q_position in {32, 25, 28}:
            return k_position == 7
        elif q_position in {30, 47}:
            return k_position == 8
        elif q_position in {48, 49, 34}:
            return k_position == 4
        elif q_position in {40}:
            return k_position == 17
        elif q_position in {41, 43}:
            return k_position == 46
        elif q_position in {44}:
            return k_position == 12

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 33, 34, 17, 20, 23, 25, 26, 29, 31}:
            return k_position == 2
        elif q_position in {1}:
            return k_position == 22
        elif q_position in {41, 2}:
            return k_position == 35
        elif q_position in {3}:
            return k_position == 30
        elif q_position in {16, 4, 14, 15}:
            return k_position == 21
        elif q_position in {5}:
            return k_position == 33
        elif q_position in {9, 6}:
            return k_position == 29
        elif q_position in {7}:
            return k_position == 23
        elif q_position in {8}:
            return k_position == 31
        elif q_position in {10}:
            return k_position == 28
        elif q_position in {11}:
            return k_position == 17
        elif q_position in {12}:
            return k_position == 18
        elif q_position in {18, 13}:
            return k_position == 20
        elif q_position in {35, 36, 37, 38, 39, 19, 21, 30}:
            return k_position == 1
        elif q_position in {22}:
            return k_position == 39
        elif q_position in {24, 32, 28}:
            return k_position == 3
        elif q_position in {27}:
            return k_position == 4
        elif q_position in {40, 49}:
            return k_position == 5
        elif q_position in {42}:
            return k_position == 43
        elif q_position in {43}:
            return k_position == 42
        elif q_position in {44}:
            return k_position == 25
        elif q_position in {45}:
            return k_position == 47
        elif q_position in {46}:
            return k_position == 19
        elif q_position in {47}:
            return k_position == 14
        elif q_position in {48}:
            return k_position == 45

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0, 37, 18, 22, 24}:
            return k_position == 2
        elif q_position in {1}:
            return k_position == 11
        elif q_position in {2}:
            return k_position == 18
        elif q_position in {3}:
            return k_position == 14
        elif q_position in {4, 6, 38, 8, 11, 13, 23, 25}:
            return k_position == 1
        elif q_position in {5}:
            return k_position == 31
        elif q_position in {7}:
            return k_position == 10
        elif q_position in {36, 9, 46, 49, 19, 21, 27}:
            return k_position == 3
        elif q_position in {10}:
            return k_position == 19
        elif q_position in {12}:
            return k_position == 24
        elif q_position in {14}:
            return k_position == 22
        elif q_position in {35, 15, 16, 17, 20}:
            return k_position == 4
        elif q_position in {26, 30}:
            return k_position == 7
        elif q_position in {34, 28}:
            return k_position == 5
        elif q_position in {29, 31}:
            return k_position == 8
        elif q_position in {32, 33}:
            return k_position == 6
        elif q_position in {39}:
            return k_position == 13
        elif q_position in {40}:
            return k_position == 34
        elif q_position in {48, 41, 44}:
            return k_position == 20
        elif q_position in {42}:
            return k_position == 47
        elif q_position in {43}:
            return k_position == 26
        elif q_position in {45}:
            return k_position == 38
        elif q_position in {47}:
            return k_position == 32

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 32, 34, 37, 5, 38}:
            return k_position == 1
        elif q_position in {1, 42, 45, 14}:
            return k_position == 24
        elif q_position in {2}:
            return k_position == 32
        elif q_position in {3, 4}:
            return k_position == 35
        elif q_position in {13, 6}:
            return k_position == 21
        elif q_position in {7}:
            return k_position == 30
        elif q_position in {8}:
            return k_position == 25
        elif q_position in {9}:
            return k_position == 14
        elif q_position in {10}:
            return k_position == 27
        elif q_position in {11, 12}:
            return k_position == 23
        elif q_position in {15, 17, 20, 29, 30}:
            return k_position == 3
        elif q_position in {16}:
            return k_position == 20
        elif q_position in {18}:
            return k_position == 0
        elif q_position in {19, 47}:
            return k_position == 34
        elif q_position in {27, 21, 39}:
            return k_position == 39
        elif q_position in {33, 35, 48, 22, 25}:
            return k_position == 4
        elif q_position in {23}:
            return k_position == 38
        elif q_position in {24}:
            return k_position == 37
        elif q_position in {26}:
            return k_position == 5
        elif q_position in {28, 36}:
            return k_position == 2
        elif q_position in {31}:
            return k_position == 6
        elif q_position in {40, 41}:
            return k_position == 47
        elif q_position in {43}:
            return k_position == 26
        elif q_position in {44}:
            return k_position == 36
        elif q_position in {46}:
            return k_position == 42
        elif q_position in {49}:
            return k_position == 9

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(q_position, k_position):
        if q_position in {0, 38, 39, 10, 12, 14, 17, 18, 20, 24, 26, 27, 28, 29}:
            return k_position == 1
        elif q_position in {1, 5, 7}:
            return k_position == 28
        elif q_position in {2}:
            return k_position == 26
        elif q_position in {41, 3}:
            return k_position == 21
        elif q_position in {4}:
            return k_position == 34
        elif q_position in {6}:
            return k_position == 32
        elif q_position in {8}:
            return k_position == 24
        elif q_position in {9}:
            return k_position == 30
        elif q_position in {35, 36, 37, 11, 13, 15, 16, 19, 21, 30}:
            return k_position == 2
        elif q_position in {33, 34, 44, 47, 22, 23, 25}:
            return k_position == 3
        elif q_position in {32, 31}:
            return k_position == 4
        elif q_position in {40}:
            return k_position == 37
        elif q_position in {42}:
            return k_position == 14
        elif q_position in {43}:
            return k_position == 29
        elif q_position in {45}:
            return k_position == 47
        elif q_position in {48, 46}:
            return k_position == 40
        elif q_position in {49}:
            return k_position == 27

    attn_0_4_pattern = select_closest(positions, positions, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(q_position, k_position):
        if q_position in {0, 41, 35}:
            return k_position == 3
        elif q_position in {1, 47}:
            return k_position == 10
        elif q_position in {2}:
            return k_position == 20
        elif q_position in {3}:
            return k_position == 18
        elif q_position in {11, 4}:
            return k_position == 28
        elif q_position in {5, 46}:
            return k_position == 33
        elif q_position in {20, 6}:
            return k_position == 9
        elif q_position in {45, 7}:
            return k_position == 12
        elif q_position in {8}:
            return k_position == 29
        elif q_position in {33, 36, 37, 38, 9, 31}:
            return k_position == 1
        elif q_position in {10}:
            return k_position == 25
        elif q_position in {12, 13, 14, 15, 16, 17, 19}:
            return k_position == 0
        elif q_position in {34, 48, 18, 26, 28, 30}:
            return k_position == 4
        elif q_position in {21}:
            return k_position == 8
        elif q_position in {25, 27, 29, 22}:
            return k_position == 5
        elif q_position in {23}:
            return k_position == 39
        elif q_position in {24}:
            return k_position == 6
        elif q_position in {32}:
            return k_position == 2
        elif q_position in {39}:
            return k_position == 35
        elif q_position in {40}:
            return k_position == 26
        elif q_position in {42}:
            return k_position == 47
        elif q_position in {43}:
            return k_position == 13
        elif q_position in {44}:
            return k_position == 46
        elif q_position in {49}:
            return k_position == 15

    attn_0_5_pattern = select_closest(positions, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(q_token, k_token):
        if q_token in {"</s>", "0", "<s>", "4"}:
            return k_token == "1"
        elif q_token in {"2", "1", "3"}:
            return k_token == "0"

    attn_0_6_pattern = select_closest(tokens, tokens, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(q_position, k_position):
        if q_position in {0, 37}:
            return k_position == 2
        elif q_position in {1, 28, 29, 23}:
            return k_position == 9
        elif q_position in {2, 20, 15}:
            return k_position == 7
        elif q_position in {3, 17, 22, 25, 27}:
            return k_position == 11
        elif q_position in {4}:
            return k_position == 18
        elif q_position in {5}:
            return k_position == 34
        elif q_position in {45, 6}:
            return k_position == 30
        elif q_position in {7, 41, 42, 12, 44, 46}:
            return k_position == 4
        elif q_position in {8}:
            return k_position == 17
        elif q_position in {9}:
            return k_position == 23
        elif q_position in {10}:
            return k_position == 20
        elif q_position in {11}:
            return k_position == 26
        elif q_position in {13}:
            return k_position == 25
        elif q_position in {14}:
            return k_position == 8
        elif q_position in {16, 19}:
            return k_position == 6
        elif q_position in {32, 33, 18, 30, 31}:
            return k_position == 5
        elif q_position in {24, 21}:
            return k_position == 10
        elif q_position in {26}:
            return k_position == 12
        elif q_position in {48, 49, 34, 47}:
            return k_position == 3
        elif q_position in {35, 36, 38}:
            return k_position == 1
        elif q_position in {39}:
            return k_position == 35
        elif q_position in {40}:
            return k_position == 24
        elif q_position in {43}:
            return k_position == 31

    attn_0_7_pattern = select_closest(positions, positions, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_position, k_position):
        if q_position in {0, 8, 46, 47, 28}:
            return k_position == 49
        elif q_position in {38, 1, 37, 17}:
            return k_position == 44
        elif q_position in {2, 35, 41, 22, 23}:
            return k_position == 47
        elif q_position in {3, 6, 7}:
            return k_position == 45
        elif q_position in {9, 4, 20}:
            return k_position == 43
        elif q_position in {32, 26, 34, 5}:
            return k_position == 40
        elif q_position in {10, 11, 45, 14, 18}:
            return k_position == 48
        elif q_position in {12}:
            return k_position == 15
        elif q_position in {27, 13}:
            return k_position == 10
        elif q_position in {42, 31, 39, 15}:
            return k_position == 46
        elif q_position in {33, 36, 16, 19, 29, 30}:
            return k_position == 41
        elif q_position in {21}:
            return k_position == 29
        elif q_position in {24}:
            return k_position == 27
        elif q_position in {25}:
            return k_position == 17
        elif q_position in {40}:
            return k_position == 39
        elif q_position in {43}:
            return k_position == 37
        elif q_position in {44}:
            return k_position == 34
        elif q_position in {48}:
            return k_position == 35
        elif q_position in {49}:
            return k_position == 38

    num_attn_0_0_pattern = select(positions, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(token, position):
        if token in {"0"}:
            return position == 43
        elif token in {"1"}:
            return position == 40
        elif token in {"2", "3"}:
            return position == 41
        elif token in {"4"}:
            return position == 46
        elif token in {"</s>"}:
            return position == 44
        elif token in {"<s>"}:
            return position == 42

    num_attn_0_1_pattern = select(positions, tokens, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(token, position):
        if token in {"0"}:
            return position == 49
        elif token in {"1"}:
            return position == 47
        elif token in {"2", "4"}:
            return position == 46
        elif token in {"3"}:
            return position == 36
        elif token in {"</s>"}:
            return position == 42
        elif token in {"<s>"}:
            return position == 37

    num_attn_0_2_pattern = select(positions, tokens, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(token, position):
        if token in {"0", "3"}:
            return position == 44
        elif token in {"1"}:
            return position == 49
        elif token in {"2", "<s>"}:
            return position == 46
        elif token in {"4"}:
            return position == 24
        elif token in {"</s>"}:
            return position == 41

    num_attn_0_3_pattern = select(positions, tokens, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(token, position):
        if token in {"0"}:
            return position == 45
        elif token in {"1"}:
            return position == 40
        elif token in {"2"}:
            return position == 44
        elif token in {"3"}:
            return position == 12
        elif token in {"4"}:
            return position == 43
        elif token in {"</s>"}:
            return position == 48
        elif token in {"<s>"}:
            return position == 13

    num_attn_0_4_pattern = select(positions, tokens, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(token, position):
        if token in {"0"}:
            return position == 46
        elif token in {"1"}:
            return position == 47
        elif token in {"2"}:
            return position == 48
        elif token in {"3"}:
            return position == 43
        elif token in {"4"}:
            return position == 42
        elif token in {"</s>"}:
            return position == 44
        elif token in {"<s>"}:
            return position == 11

    num_attn_0_5_pattern = select(positions, tokens, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(q_token, k_token):
        if q_token in {"2", "0", "</s>", "3", "4", "1", "<s>"}:
            return k_token == ""

    num_attn_0_6_pattern = select(tokens, tokens, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(q_token, k_token):
        if q_token in {"2", "0", "</s>", "3", "4", "1", "<s>"}:
            return k_token == ""

    num_attn_0_7_pattern = select(tokens, tokens, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(token, attn_0_6_output):
        key = (token, attn_0_6_output)
        if key in {
            ("0", "3"),
            ("1", "3"),
            ("2", "3"),
            ("3", "3"),
            ("4", "3"),
            ("</s>", "0"),
            ("</s>", "1"),
            ("</s>", "2"),
            ("</s>", "3"),
            ("</s>", "4"),
            ("</s>", "</s>"),
            ("</s>", "<s>"),
            ("<s>", "3"),
        }:
            return 3
        return 5

    mlp_0_0_outputs = [mlp_0_0(k0, k1) for k0, k1 in zip(tokens, attn_0_6_outputs)]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_6_output, attn_0_7_output):
        key = (attn_0_6_output, attn_0_7_output)
        return 5

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_6_outputs, attn_0_7_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_2_output, num_attn_0_6_output):
        key = (num_attn_0_2_output, num_attn_0_6_output)
        return 7

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_0_6_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_7_output, num_attn_0_3_output):
        key = (num_attn_0_7_output, num_attn_0_3_output)
        return 2

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_7_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(attn_0_1_output, token):
        if attn_0_1_output in {"</s>", "0", "<s>"}:
            return token == "4"
        elif attn_0_1_output in {"2", "1"}:
            return token == "<s>"
        elif attn_0_1_output in {"3", "4"}:
            return token == ""

    attn_1_0_pattern = select_closest(tokens, attn_0_1_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_6_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(attn_0_4_output, token):
        if attn_0_4_output in {"0"}:
            return token == "0"
        elif attn_0_4_output in {"1"}:
            return token == "1"
        elif attn_0_4_output in {"2", "3", "4"}:
            return token == ""
        elif attn_0_4_output in {"</s>"}:
            return token == "</s>"
        elif attn_0_4_output in {"<s>"}:
            return token == "2"

    attn_1_1_pattern = select_closest(tokens, attn_0_4_outputs, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_1_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(attn_0_3_output, position):
        if attn_0_3_output in {"1", "0"}:
            return position == 10
        elif attn_0_3_output in {"2", "<s>"}:
            return position == 7
        elif attn_0_3_output in {"3"}:
            return position == 11
        elif attn_0_3_output in {"4"}:
            return position == 8
        elif attn_0_3_output in {"</s>"}:
            return position == 12

    attn_1_2_pattern = select_closest(positions, attn_0_3_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, tokens)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(attn_0_1_output, token):
        if attn_0_1_output in {"0"}:
            return token == "4"
        elif attn_0_1_output in {"2", "</s>", "4", "1", "<s>"}:
            return token == "3"
        elif attn_0_1_output in {"3"}:
            return token == "2"

    attn_1_3_pattern = select_closest(tokens, attn_0_1_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, tokens)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(attn_0_1_output, token):
        if attn_0_1_output in {"0", "3", "4"}:
            return token == "0"
        elif attn_0_1_output in {"1"}:
            return token == "</s>"
        elif attn_0_1_output in {"2"}:
            return token == "1"
        elif attn_0_1_output in {"</s>", "<s>"}:
            return token == "4"

    attn_1_4_pattern = select_closest(tokens, attn_0_1_outputs, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, tokens)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(attn_0_3_output, attn_0_6_output):
        if attn_0_3_output in {"0"}:
            return attn_0_6_output == ""
        elif attn_0_3_output in {"1"}:
            return attn_0_6_output == "</s>"
        elif attn_0_3_output in {"2"}:
            return attn_0_6_output == "0"
        elif attn_0_3_output in {"<s>", "3"}:
            return attn_0_6_output == "2"
        elif attn_0_3_output in {"4"}:
            return attn_0_6_output == "<s>"
        elif attn_0_3_output in {"</s>"}:
            return attn_0_6_output == "4"

    attn_1_5_pattern = select_closest(attn_0_6_outputs, attn_0_3_outputs, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, tokens)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(mlp_0_0_output, attn_0_5_output):
        if mlp_0_0_output in {
            0,
            1,
            6,
            7,
            8,
            9,
            10,
            11,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
        }:
            return attn_0_5_output == ""
        elif mlp_0_0_output in {2, 3, 4, 5, 12}:
            return attn_0_5_output == "<s>"

    attn_1_6_pattern = select_closest(attn_0_5_outputs, mlp_0_0_outputs, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, tokens)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(attn_0_5_output, token):
        if attn_0_5_output in {"0", "<s>"}:
            return token == ""
        elif attn_0_5_output in {"1"}:
            return token == "1"
        elif attn_0_5_output in {"2", "</s>"}:
            return token == "2"
        elif attn_0_5_output in {"3"}:
            return token == "3"
        elif attn_0_5_output in {"4"}:
            return token == "4"

    attn_1_7_pattern = select_closest(tokens, attn_0_5_outputs, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, tokens)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(attn_0_1_output, attn_0_0_output):
        if attn_0_1_output in {"2", "0", "3", "4", "1", "<s>"}:
            return attn_0_0_output == ""
        elif attn_0_1_output in {"</s>"}:
            return attn_0_0_output == "<pad>"

    num_attn_1_0_pattern = select(attn_0_0_outputs, attn_0_1_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_3_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(attn_0_4_output, attn_0_3_output):
        if attn_0_4_output in {"2", "0", "</s>", "3", "1"}:
            return attn_0_3_output == ""
        elif attn_0_4_output in {"4"}:
            return attn_0_3_output == "</s>"
        elif attn_0_4_output in {"<s>"}:
            return attn_0_3_output == "4"

    num_attn_1_1_pattern = select(attn_0_3_outputs, attn_0_4_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_5_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(num_mlp_0_0_output, mlp_0_0_output):
        if num_mlp_0_0_output in {0, 38, 8, 41, 24}:
            return mlp_0_0_output == 21
        elif num_mlp_0_0_output in {1}:
            return mlp_0_0_output == 23
        elif num_mlp_0_0_output in {2}:
            return mlp_0_0_output == 29
        elif num_mlp_0_0_output in {40, 3}:
            return mlp_0_0_output == 15
        elif num_mlp_0_0_output in {4}:
            return mlp_0_0_output == 47
        elif num_mlp_0_0_output in {5, 23}:
            return mlp_0_0_output == 7
        elif num_mlp_0_0_output in {6}:
            return mlp_0_0_output == 41
        elif num_mlp_0_0_output in {7}:
            return mlp_0_0_output == 43
        elif num_mlp_0_0_output in {9}:
            return mlp_0_0_output == 33
        elif num_mlp_0_0_output in {10, 44, 30}:
            return mlp_0_0_output == 18
        elif num_mlp_0_0_output in {11, 46}:
            return mlp_0_0_output == 6
        elif num_mlp_0_0_output in {28, 19, 12, 20}:
            return mlp_0_0_output == 13
        elif num_mlp_0_0_output in {32, 49, 37, 13}:
            return mlp_0_0_output == 46
        elif num_mlp_0_0_output in {14}:
            return mlp_0_0_output == 42
        elif num_mlp_0_0_output in {43, 15}:
            return mlp_0_0_output == 20
        elif num_mlp_0_0_output in {16}:
            return mlp_0_0_output == 30
        elif num_mlp_0_0_output in {17}:
            return mlp_0_0_output == 14
        elif num_mlp_0_0_output in {18, 29}:
            return mlp_0_0_output == 34
        elif num_mlp_0_0_output in {33, 21, 47}:
            return mlp_0_0_output == 8
        elif num_mlp_0_0_output in {22}:
            return mlp_0_0_output == 48
        elif num_mlp_0_0_output in {25}:
            return mlp_0_0_output == 25
        elif num_mlp_0_0_output in {26, 42}:
            return mlp_0_0_output == 24
        elif num_mlp_0_0_output in {27}:
            return mlp_0_0_output == 11
        elif num_mlp_0_0_output in {31}:
            return mlp_0_0_output == 39
        elif num_mlp_0_0_output in {34}:
            return mlp_0_0_output == 19
        elif num_mlp_0_0_output in {48, 35}:
            return mlp_0_0_output == 49
        elif num_mlp_0_0_output in {36}:
            return mlp_0_0_output == 40
        elif num_mlp_0_0_output in {39}:
            return mlp_0_0_output == 44
        elif num_mlp_0_0_output in {45}:
            return mlp_0_0_output == 10

    num_attn_1_2_pattern = select(
        mlp_0_0_outputs, num_mlp_0_0_outputs, num_predicate_1_2
    )
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_5_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(attn_0_4_output, attn_0_3_output):
        if attn_0_4_output in {"1", "0", "3", "4"}:
            return attn_0_3_output == ""
        elif attn_0_4_output in {"2", "</s>", "<s>"}:
            return attn_0_3_output == "2"

    num_attn_1_3_pattern = select(attn_0_3_outputs, attn_0_4_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_5_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(num_mlp_0_0_output, num_mlp_0_1_output):
        if num_mlp_0_0_output in {0}:
            return num_mlp_0_1_output == 11
        elif num_mlp_0_0_output in {1, 35, 33}:
            return num_mlp_0_1_output == 29
        elif num_mlp_0_0_output in {2}:
            return num_mlp_0_1_output == 14
        elif num_mlp_0_0_output in {3, 12}:
            return num_mlp_0_1_output == 17
        elif num_mlp_0_0_output in {18, 4}:
            return num_mlp_0_1_output == 25
        elif num_mlp_0_0_output in {37, 5}:
            return num_mlp_0_1_output == 8
        elif num_mlp_0_0_output in {36, 6}:
            return num_mlp_0_1_output == 43
        elif num_mlp_0_0_output in {48, 10, 7}:
            return num_mlp_0_1_output == 21
        elif num_mlp_0_0_output in {8, 38}:
            return num_mlp_0_1_output == 31
        elif num_mlp_0_0_output in {9}:
            return num_mlp_0_1_output == 26
        elif num_mlp_0_0_output in {11, 44}:
            return num_mlp_0_1_output == 10
        elif num_mlp_0_0_output in {21, 13}:
            return num_mlp_0_1_output == 12
        elif num_mlp_0_0_output in {14}:
            return num_mlp_0_1_output == 15
        elif num_mlp_0_0_output in {23, 15}:
            return num_mlp_0_1_output == 9
        elif num_mlp_0_0_output in {16, 42}:
            return num_mlp_0_1_output == 13
        elif num_mlp_0_0_output in {24, 17}:
            return num_mlp_0_1_output == 44
        elif num_mlp_0_0_output in {32, 19}:
            return num_mlp_0_1_output == 28
        elif num_mlp_0_0_output in {20, 47}:
            return num_mlp_0_1_output == 49
        elif num_mlp_0_0_output in {22}:
            return num_mlp_0_1_output == 27
        elif num_mlp_0_0_output in {25, 27}:
            return num_mlp_0_1_output == 36
        elif num_mlp_0_0_output in {26, 46}:
            return num_mlp_0_1_output == 0
        elif num_mlp_0_0_output in {43, 28}:
            return num_mlp_0_1_output == 20
        elif num_mlp_0_0_output in {34, 29}:
            return num_mlp_0_1_output == 33
        elif num_mlp_0_0_output in {30}:
            return num_mlp_0_1_output == 19
        elif num_mlp_0_0_output in {40, 31}:
            return num_mlp_0_1_output == 37
        elif num_mlp_0_0_output in {39}:
            return num_mlp_0_1_output == 39
        elif num_mlp_0_0_output in {41}:
            return num_mlp_0_1_output == 46
        elif num_mlp_0_0_output in {49, 45}:
            return num_mlp_0_1_output == 23

    num_attn_1_4_pattern = select(
        num_mlp_0_1_outputs, num_mlp_0_0_outputs, num_predicate_1_4
    )
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_7_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(attn_0_4_output, attn_0_3_output):
        if attn_0_4_output in {"2", "</s>", "0", "3", "4"}:
            return attn_0_3_output == ""
        elif attn_0_4_output in {"1"}:
            return attn_0_3_output == "<pad>"
        elif attn_0_4_output in {"<s>"}:
            return attn_0_3_output == "1"

    num_attn_1_5_pattern = select(attn_0_3_outputs, attn_0_4_outputs, num_predicate_1_5)
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_5_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(attn_0_2_output, num_mlp_0_0_output):
        if attn_0_2_output in {"0"}:
            return num_mlp_0_0_output == 33
        elif attn_0_2_output in {"1"}:
            return num_mlp_0_0_output == 12
        elif attn_0_2_output in {"2", "<s>"}:
            return num_mlp_0_0_output == 49
        elif attn_0_2_output in {"3"}:
            return num_mlp_0_0_output == 45
        elif attn_0_2_output in {"4"}:
            return num_mlp_0_0_output == 36
        elif attn_0_2_output in {"</s>"}:
            return num_mlp_0_0_output == 9

    num_attn_1_6_pattern = select(
        num_mlp_0_0_outputs, attn_0_2_outputs, num_predicate_1_6
    )
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_1_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(num_mlp_0_1_output, mlp_0_1_output):
        if num_mlp_0_1_output in {0, 2, 12}:
            return mlp_0_1_output == 25
        elif num_mlp_0_1_output in {1}:
            return mlp_0_1_output == 37
        elif num_mlp_0_1_output in {3}:
            return mlp_0_1_output == 34
        elif num_mlp_0_1_output in {40, 4, 21}:
            return mlp_0_1_output == 18
        elif num_mlp_0_1_output in {20, 5}:
            return mlp_0_1_output == 14
        elif num_mlp_0_1_output in {43, 6}:
            return mlp_0_1_output == 11
        elif num_mlp_0_1_output in {7}:
            return mlp_0_1_output == 12
        elif num_mlp_0_1_output in {8, 10, 46}:
            return mlp_0_1_output == 46
        elif num_mlp_0_1_output in {9, 37}:
            return mlp_0_1_output == 9
        elif num_mlp_0_1_output in {49, 11}:
            return mlp_0_1_output == 45
        elif num_mlp_0_1_output in {13}:
            return mlp_0_1_output == 35
        elif num_mlp_0_1_output in {45, 14, 22}:
            return mlp_0_1_output == 49
        elif num_mlp_0_1_output in {15}:
            return mlp_0_1_output == 30
        elif num_mlp_0_1_output in {16, 26}:
            return mlp_0_1_output == 8
        elif num_mlp_0_1_output in {17, 41}:
            return mlp_0_1_output == 33
        elif num_mlp_0_1_output in {48, 18, 19}:
            return mlp_0_1_output == 15
        elif num_mlp_0_1_output in {23}:
            return mlp_0_1_output == 29
        elif num_mlp_0_1_output in {24, 44, 30}:
            return mlp_0_1_output == 24
        elif num_mlp_0_1_output in {25}:
            return mlp_0_1_output == 17
        elif num_mlp_0_1_output in {27}:
            return mlp_0_1_output == 32
        elif num_mlp_0_1_output in {28, 38}:
            return mlp_0_1_output == 10
        elif num_mlp_0_1_output in {42, 29}:
            return mlp_0_1_output == 31
        elif num_mlp_0_1_output in {31}:
            return mlp_0_1_output == 42
        elif num_mlp_0_1_output in {32}:
            return mlp_0_1_output == 36
        elif num_mlp_0_1_output in {33, 47}:
            return mlp_0_1_output == 0
        elif num_mlp_0_1_output in {34}:
            return mlp_0_1_output == 40
        elif num_mlp_0_1_output in {35}:
            return mlp_0_1_output == 21
        elif num_mlp_0_1_output in {36}:
            return mlp_0_1_output == 27
        elif num_mlp_0_1_output in {39}:
            return mlp_0_1_output == 2

    num_attn_1_7_pattern = select(
        mlp_0_1_outputs, num_mlp_0_1_outputs, num_predicate_1_7
    )
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_5_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_0_3_output, attn_0_2_output):
        key = (attn_0_3_output, attn_0_2_output)
        if key in {("3", "<s>"), ("4", "<s>"), ("</s>", "<s>"), ("<s>", "<s>")}:
            return 19
        return 3

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_0_3_outputs, attn_0_2_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_0_5_output, attn_1_6_output):
        key = (attn_0_5_output, attn_1_6_output)
        return 22

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_0_5_outputs, attn_1_6_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_0_output, num_attn_1_5_output):
        key = (num_attn_1_0_output, num_attn_1_5_output)
        return 3

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_1_5_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_7_output):
        key = num_attn_1_7_output
        return 30

    num_mlp_1_1_outputs = [num_mlp_1_1(k0) for k0 in num_attn_1_7_outputs]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(q_attn_0_1_output, k_attn_0_1_output):
        if q_attn_0_1_output in {"1", "0", "</s>"}:
            return k_attn_0_1_output == ""
        elif q_attn_0_1_output in {"2", "<s>"}:
            return k_attn_0_1_output == "4"
        elif q_attn_0_1_output in {"3"}:
            return k_attn_0_1_output == "</s>"
        elif q_attn_0_1_output in {"4"}:
            return k_attn_0_1_output == "<s>"

    attn_2_0_pattern = select_closest(attn_0_1_outputs, attn_0_1_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_4_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(attn_0_1_output, attn_0_0_output):
        if attn_0_1_output in {"0"}:
            return attn_0_0_output == "1"
        elif attn_0_1_output in {"1", "<s>"}:
            return attn_0_0_output == "4"
        elif attn_0_1_output in {"2"}:
            return attn_0_0_output == "3"
        elif attn_0_1_output in {"</s>", "3", "4"}:
            return attn_0_0_output == "0"

    attn_2_1_pattern = select_closest(attn_0_0_outputs, attn_0_1_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_6_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(attn_0_2_output, attn_1_5_output):
        if attn_0_2_output in {"0"}:
            return attn_1_5_output == "4"
        elif attn_0_2_output in {"1", "</s>", "3"}:
            return attn_1_5_output == ""
        elif attn_0_2_output in {"2"}:
            return attn_1_5_output == "0"
        elif attn_0_2_output in {"4"}:
            return attn_1_5_output == "2"
        elif attn_0_2_output in {"<s>"}:
            return attn_1_5_output == "1"

    attn_2_2_pattern = select_closest(attn_1_5_outputs, attn_0_2_outputs, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_5_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(attn_0_4_output, mlp_0_1_output):
        if attn_0_4_output in {"0"}:
            return mlp_0_1_output == 6
        elif attn_0_4_output in {"1"}:
            return mlp_0_1_output == 46
        elif attn_0_4_output in {"2"}:
            return mlp_0_1_output == 16
        elif attn_0_4_output in {"</s>", "3"}:
            return mlp_0_1_output == 1
        elif attn_0_4_output in {"4"}:
            return mlp_0_1_output == 39
        elif attn_0_4_output in {"<s>"}:
            return mlp_0_1_output == 2

    attn_2_3_pattern = select_closest(mlp_0_1_outputs, attn_0_4_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_2_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(attn_0_1_output, mlp_0_0_output):
        if attn_0_1_output in {"0"}:
            return mlp_0_0_output == 0
        elif attn_0_1_output in {"1"}:
            return mlp_0_0_output == 39
        elif attn_0_1_output in {"2"}:
            return mlp_0_0_output == 41
        elif attn_0_1_output in {"3"}:
            return mlp_0_0_output == 1
        elif attn_0_1_output in {"4"}:
            return mlp_0_0_output == 4
        elif attn_0_1_output in {"</s>"}:
            return mlp_0_0_output == 13
        elif attn_0_1_output in {"<s>"}:
            return mlp_0_0_output == 5

    attn_2_4_pattern = select_closest(mlp_0_0_outputs, attn_0_1_outputs, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_1_6_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(attn_0_1_output, token):
        if attn_0_1_output in {"</s>", "0"}:
            return token == "4"
        elif attn_0_1_output in {"1"}:
            return token == "3"
        elif attn_0_1_output in {"2", "<s>"}:
            return token == ""
        elif attn_0_1_output in {"3"}:
            return token == "0"
        elif attn_0_1_output in {"4"}:
            return token == "</s>"

    attn_2_5_pattern = select_closest(tokens, attn_0_1_outputs, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_1_7_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(attn_0_1_output, attn_1_3_output):
        if attn_0_1_output in {"2", "0", "<s>"}:
            return attn_1_3_output == "4"
        elif attn_0_1_output in {"1", "</s>"}:
            return attn_1_3_output == ""
        elif attn_0_1_output in {"3"}:
            return attn_1_3_output == "3"
        elif attn_0_1_output in {"4"}:
            return attn_1_3_output == "<s>"

    attn_2_6_pattern = select_closest(attn_1_3_outputs, attn_0_1_outputs, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, attn_1_3_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(attn_0_4_output, mlp_0_1_output):
        if attn_0_4_output in {"0"}:
            return mlp_0_1_output == 6
        elif attn_0_4_output in {"1"}:
            return mlp_0_1_output == 35
        elif attn_0_4_output in {"2"}:
            return mlp_0_1_output == 4
        elif attn_0_4_output in {"3"}:
            return mlp_0_1_output == 27
        elif attn_0_4_output in {"4"}:
            return mlp_0_1_output == 7
        elif attn_0_4_output in {"</s>"}:
            return mlp_0_1_output == 2
        elif attn_0_4_output in {"<s>"}:
            return mlp_0_1_output == 5

    attn_2_7_pattern = select_closest(mlp_0_1_outputs, attn_0_4_outputs, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_1_1_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(q_attn_0_2_output, k_attn_0_2_output):
        if q_attn_0_2_output in {"0"}:
            return k_attn_0_2_output == "0"
        elif q_attn_0_2_output in {"2", "</s>", "3", "4", "1", "<s>"}:
            return k_attn_0_2_output == ""

    num_attn_2_0_pattern = select(attn_0_2_outputs, attn_0_2_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_0_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(mlp_1_1_output, attn_0_6_output):
        if mlp_1_1_output in {
            0,
            1,
            2,
            3,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
        }:
            return attn_0_6_output == ""
        elif mlp_1_1_output in {4}:
            return attn_0_6_output == "<s>"
        elif mlp_1_1_output in {24}:
            return attn_0_6_output == "<pad>"

    num_attn_2_1_pattern = select(attn_0_6_outputs, mlp_1_1_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_1_6_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_1_1_output, attn_0_4_output):
        if attn_1_1_output in {"2", "0", "</s>", "3", "4", "1", "<s>"}:
            return attn_0_4_output == ""

    num_attn_2_2_pattern = select(attn_0_4_outputs, attn_1_1_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_1_4_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(num_mlp_1_1_output, attn_0_0_output):
        if num_mlp_1_1_output in {
            0,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            33,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            49,
        }:
            return attn_0_0_output == ""
        elif num_mlp_1_1_output in {1, 34}:
            return attn_0_0_output == "<pad>"
        elif num_mlp_1_1_output in {32}:
            return attn_0_0_output == "2"
        elif num_mlp_1_1_output in {47}:
            return attn_0_0_output == "</s>"
        elif num_mlp_1_1_output in {48}:
            return attn_0_0_output == "4"

    num_attn_2_3_pattern = select(
        attn_0_0_outputs, num_mlp_1_1_outputs, num_predicate_2_3
    )
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_1_1_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(attn_0_2_output, attn_0_4_output):
        if attn_0_2_output in {"0", "</s>", "3", "1", "<s>"}:
            return attn_0_4_output == ""
        elif attn_0_2_output in {"2"}:
            return attn_0_4_output == "<pad>"
        elif attn_0_2_output in {"4"}:
            return attn_0_4_output == "4"

    num_attn_2_4_pattern = select(attn_0_4_outputs, attn_0_2_outputs, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_0_0_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(q_attn_0_1_output, k_attn_0_1_output):
        if q_attn_0_1_output in {"2", "0", "</s>", "3", "4", "1", "<s>"}:
            return k_attn_0_1_output == ""

    num_attn_2_5_pattern = select(attn_0_1_outputs, attn_0_1_outputs, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_1_2_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(attn_1_2_output, attn_0_1_output):
        if attn_1_2_output in {"2", "0", "</s>", "3", "4", "1", "<s>"}:
            return attn_0_1_output == ""

    num_attn_2_6_pattern = select(attn_0_1_outputs, attn_1_2_outputs, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_1_4_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(attn_0_2_output, attn_0_4_output):
        if attn_0_2_output in {"0", "</s>", "4", "1", "<s>"}:
            return attn_0_4_output == ""
        elif attn_0_2_output in {"2"}:
            return attn_0_4_output == "<pad>"
        elif attn_0_2_output in {"3"}:
            return attn_0_4_output == "3"

    num_attn_2_7_pattern = select(attn_0_4_outputs, attn_0_2_outputs, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_0_0_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_1_5_output, attn_2_3_output):
        key = (attn_1_5_output, attn_2_3_output)
        return 39

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_1_5_outputs, attn_2_3_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(num_mlp_0_1_output, attn_2_1_output):
        key = (num_mlp_0_1_output, attn_2_1_output)
        return 36

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(num_mlp_0_1_outputs, attn_2_1_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_2_3_output, num_attn_1_5_output):
        key = (num_attn_2_3_output, num_attn_1_5_output)
        return 14

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_2_3_outputs, num_attn_1_5_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_5_output, num_attn_2_0_output):
        key = (num_attn_1_5_output, num_attn_2_0_output)
        return 24

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_1_5_outputs, num_attn_2_0_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    feature_logits = pd.concat(
        [
            df.reset_index()
            for df in [
                token_scores,
                position_scores,
                attn_0_0_output_scores,
                attn_0_1_output_scores,
                attn_0_2_output_scores,
                attn_0_3_output_scores,
                attn_0_4_output_scores,
                attn_0_5_output_scores,
                attn_0_6_output_scores,
                attn_0_7_output_scores,
                mlp_0_0_output_scores,
                mlp_0_1_output_scores,
                num_mlp_0_0_output_scores,
                num_mlp_0_1_output_scores,
                attn_1_0_output_scores,
                attn_1_1_output_scores,
                attn_1_2_output_scores,
                attn_1_3_output_scores,
                attn_1_4_output_scores,
                attn_1_5_output_scores,
                attn_1_6_output_scores,
                attn_1_7_output_scores,
                mlp_1_0_output_scores,
                mlp_1_1_output_scores,
                num_mlp_1_0_output_scores,
                num_mlp_1_1_output_scores,
                attn_2_0_output_scores,
                attn_2_1_output_scores,
                attn_2_2_output_scores,
                attn_2_3_output_scores,
                attn_2_4_output_scores,
                attn_2_5_output_scores,
                attn_2_6_output_scores,
                attn_2_7_output_scores,
                mlp_2_0_output_scores,
                mlp_2_1_output_scores,
                num_mlp_2_0_output_scores,
                num_mlp_2_1_output_scores,
                one_scores,
                num_attn_0_0_output_scores,
                num_attn_0_1_output_scores,
                num_attn_0_2_output_scores,
                num_attn_0_3_output_scores,
                num_attn_0_4_output_scores,
                num_attn_0_5_output_scores,
                num_attn_0_6_output_scores,
                num_attn_0_7_output_scores,
                num_attn_1_0_output_scores,
                num_attn_1_1_output_scores,
                num_attn_1_2_output_scores,
                num_attn_1_3_output_scores,
                num_attn_1_4_output_scores,
                num_attn_1_5_output_scores,
                num_attn_1_6_output_scores,
                num_attn_1_7_output_scores,
                num_attn_2_0_output_scores,
                num_attn_2_1_output_scores,
                num_attn_2_2_output_scores,
                num_attn_2_3_output_scores,
                num_attn_2_4_output_scores,
                num_attn_2_5_output_scores,
                num_attn_2_6_output_scores,
                num_attn_2_7_output_scores,
            ]
        ]
    )
    logits = feature_logits.groupby(level=0).sum(numeric_only=True).to_numpy()
    classes = classifier_weights.columns.to_numpy()
    predictions = classes[logits.argmax(-1)]
    if tokens[0] == "<s>":
        predictions[0] = "<s>"
    if tokens[-1] == "</s>":
        predictions[-1] = "</s>"
    return predictions.tolist()


print(run(["<s>", "3", "</s>"]))
