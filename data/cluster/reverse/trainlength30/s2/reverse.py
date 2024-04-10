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
        "output/length/rasp/reverse/trainlength30/s2/reverse_weights.csv",
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
        if q_position in {0, 20, 23}:
            return k_position == 5
        elif q_position in {1}:
            return k_position == 10
        elif q_position in {2, 30}:
            return k_position == 21
        elif q_position in {9, 3, 7}:
            return k_position == 20
        elif q_position in {4}:
            return k_position == 13
        elif q_position in {8, 5}:
            return k_position == 12
        elif q_position in {11, 6}:
            return k_position == 15
        elif q_position in {10, 12, 14, 15, 17}:
            return k_position == 3
        elif q_position in {13}:
            return k_position == 0
        elif q_position in {16}:
            return k_position == 29
        elif q_position in {18, 22, 26, 28, 29}:
            return k_position == 1
        elif q_position in {27, 19}:
            return k_position == 2
        elif q_position in {21}:
            return k_position == 6
        elif q_position in {24, 25, 32}:
            return k_position == 4
        elif q_position in {31}:
            return k_position == 25
        elif q_position in {33}:
            return k_position == 32
        elif q_position in {34}:
            return k_position == 34
        elif q_position in {35}:
            return k_position == 24
        elif q_position in {36}:
            return k_position == 19
        elif q_position in {37, 38}:
            return k_position == 26
        elif q_position in {39}:
            return k_position == 30

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 4
        elif q_position in {1}:
            return k_position == 15
        elif q_position in {2, 3}:
            return k_position == 14
        elif q_position in {4}:
            return k_position == 18
        elif q_position in {5}:
            return k_position == 22
        elif q_position in {6}:
            return k_position == 10
        elif q_position in {
            7,
            8,
            10,
            11,
            12,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            25,
            26,
            28,
            29,
        }:
            return k_position == 1
        elif q_position in {9, 13, 14, 15, 24, 27}:
            return k_position == 2
        elif q_position in {23}:
            return k_position == 3
        elif q_position in {30}:
            return k_position == 32
        elif q_position in {31}:
            return k_position == 28
        elif q_position in {32, 34, 35}:
            return k_position == 34
        elif q_position in {33, 39}:
            return k_position == 36
        elif q_position in {36}:
            return k_position == 24
        elif q_position in {37}:
            return k_position == 37
        elif q_position in {38}:
            return k_position == 17

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0}:
            return k_position == 7
        elif q_position in {1, 2}:
            return k_position == 22
        elif q_position in {3}:
            return k_position == 23
        elif q_position in {4}:
            return k_position == 25
        elif q_position in {5, 6, 7}:
            return k_position == 21
        elif q_position in {8, 12}:
            return k_position == 17
        elif q_position in {9}:
            return k_position == 16
        elif q_position in {10}:
            return k_position == 15
        elif q_position in {11}:
            return k_position == 18
        elif q_position in {35, 19, 13}:
            return k_position == 29
        elif q_position in {23, 14, 22}:
            return k_position == 4
        elif q_position in {15}:
            return k_position == 9
        elif q_position in {16, 24, 21}:
            return k_position == 5
        elif q_position in {17, 29, 20, 28}:
            return k_position == 1
        elif q_position in {18, 26}:
            return k_position == 3
        elif q_position in {25, 27}:
            return k_position == 2
        elif q_position in {30}:
            return k_position == 38
        elif q_position in {39, 37, 31}:
            return k_position == 37
        elif q_position in {32}:
            return k_position == 39
        elif q_position in {33}:
            return k_position == 35
        elif q_position in {34}:
            return k_position == 36
        elif q_position in {36}:
            return k_position == 32
        elif q_position in {38}:
            return k_position == 34

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 33, 32, 34, 35, 37, 38, 39, 18, 30}:
            return k_position == 4
        elif q_position in {1}:
            return k_position == 18
        elif q_position in {2}:
            return k_position == 23
        elif q_position in {3, 4, 31}:
            return k_position == 24
        elif q_position in {21, 5}:
            return k_position == 3
        elif q_position in {27, 20, 6}:
            return k_position == 2
        elif q_position in {19, 9, 11, 7}:
            return k_position == 5
        elif q_position in {8, 10}:
            return k_position == 6
        elif q_position in {12}:
            return k_position == 21
        elif q_position in {13}:
            return k_position == 9
        elif q_position in {14}:
            return k_position == 10
        elif q_position in {17, 15}:
            return k_position == 11
        elif q_position in {16}:
            return k_position == 12
        elif q_position in {24, 28, 22, 23}:
            return k_position == 1
        elif q_position in {25}:
            return k_position == 27
        elif q_position in {26}:
            return k_position == 29
        elif q_position in {29}:
            return k_position == 15
        elif q_position in {36}:
            return k_position == 14

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(q_position, k_position):
        if q_position in {0, 18, 13, 22}:
            return k_position == 29
        elif q_position in {1, 7}:
            return k_position == 19
        elif q_position in {33, 2, 37, 38}:
            return k_position == 24
        elif q_position in {34, 3}:
            return k_position == 25
        elif q_position in {4}:
            return k_position == 16
        elif q_position in {10, 5}:
            return k_position == 17
        elif q_position in {6}:
            return k_position == 20
        elif q_position in {8}:
            return k_position == 13
        elif q_position in {9}:
            return k_position == 15
        elif q_position in {19, 11}:
            return k_position == 3
        elif q_position in {25, 12, 23}:
            return k_position == 2
        elif q_position in {17, 14, 15}:
            return k_position == 5
        elif q_position in {16, 20}:
            return k_position == 6
        elif q_position in {21, 24, 26, 27, 28, 29}:
            return k_position == 1
        elif q_position in {30, 31}:
            return k_position == 33
        elif q_position in {32}:
            return k_position == 35
        elif q_position in {35}:
            return k_position == 34
        elif q_position in {36}:
            return k_position == 30
        elif q_position in {39}:
            return k_position == 4

    attn_0_4_pattern = select_closest(positions, positions, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(q_position, k_position):
        if q_position in {0, 17, 13, 30}:
            return k_position == 29
        elif q_position in {1}:
            return k_position == 27
        elif q_position in {2}:
            return k_position == 9
        elif q_position in {3}:
            return k_position == 26
        elif q_position in {4, 6}:
            return k_position == 23
        elif q_position in {5}:
            return k_position == 24
        elif q_position in {7}:
            return k_position == 10
        elif q_position in {8}:
            return k_position == 16
        elif q_position in {9}:
            return k_position == 13
        elif q_position in {10, 36}:
            return k_position == 14
        elif q_position in {11}:
            return k_position == 17
        elif q_position in {12, 15}:
            return k_position == 0
        elif q_position in {14, 23, 25, 27, 28, 29}:
            return k_position == 1
        elif q_position in {16, 24, 20, 22}:
            return k_position == 3
        elif q_position in {18, 26}:
            return k_position == 2
        elif q_position in {37, 19, 21}:
            return k_position == 4
        elif q_position in {31}:
            return k_position == 35
        elif q_position in {32}:
            return k_position == 38
        elif q_position in {33}:
            return k_position == 34
        elif q_position in {34, 35}:
            return k_position == 31
        elif q_position in {38}:
            return k_position == 25
        elif q_position in {39}:
            return k_position == 30

    attn_0_5_pattern = select_closest(positions, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(q_position, k_position):
        if q_position in {0}:
            return k_position == 3
        elif q_position in {1, 5, 14, 17}:
            return k_position == 9
        elif q_position in {18, 2, 11}:
            return k_position == 8
        elif q_position in {19, 3, 13}:
            return k_position == 6
        elif q_position in {33, 4, 24, 25, 26, 27, 28, 31}:
            return k_position == 1
        elif q_position in {6}:
            return k_position == 13
        elif q_position in {7}:
            return k_position == 16
        elif q_position in {8, 10, 38}:
            return k_position == 18
        elif q_position in {9}:
            return k_position == 17
        elif q_position in {12, 22}:
            return k_position == 5
        elif q_position in {16, 15}:
            return k_position == 10
        elif q_position in {20, 21}:
            return k_position == 7
        elif q_position in {23}:
            return k_position == 2
        elif q_position in {29, 30}:
            return k_position == 15
        elif q_position in {32}:
            return k_position == 20
        elif q_position in {34, 35}:
            return k_position == 38
        elif q_position in {36, 39}:
            return k_position == 31
        elif q_position in {37}:
            return k_position == 27

    attn_0_6_pattern = select_closest(positions, positions, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(q_position, k_position):
        if q_position in {0}:
            return k_position == 3
        elif q_position in {8, 1}:
            return k_position == 21
        elif q_position in {9, 2}:
            return k_position == 19
        elif q_position in {3, 14}:
            return k_position == 15
        elif q_position in {4}:
            return k_position == 14
        elif q_position in {5}:
            return k_position == 18
        elif q_position in {18, 6}:
            return k_position == 9
        elif q_position in {7, 11, 15, 17, 23}:
            return k_position == 6
        elif q_position in {16, 10, 19, 20}:
            return k_position == 8
        elif q_position in {12}:
            return k_position == 13
        elif q_position in {13}:
            return k_position == 5
        elif q_position in {25, 21}:
            return k_position == 2
        elif q_position in {22}:
            return k_position == 7
        elif q_position in {24, 36}:
            return k_position == 27
        elif q_position in {26, 27, 28}:
            return k_position == 1
        elif q_position in {35, 29}:
            return k_position == 12
        elif q_position in {30}:
            return k_position == 24
        elif q_position in {31}:
            return k_position == 28
        elif q_position in {32}:
            return k_position == 34
        elif q_position in {33}:
            return k_position == 17
        elif q_position in {34}:
            return k_position == 11
        elif q_position in {37}:
            return k_position == 23
        elif q_position in {38}:
            return k_position == 33
        elif q_position in {39}:
            return k_position == 31

    attn_0_7_pattern = select_closest(positions, positions, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_position, k_position):
        if q_position in {0, 11, 3, 31}:
            return k_position == 31
        elif q_position in {8, 1, 18, 7}:
            return k_position == 12
        elif q_position in {2}:
            return k_position == 9
        elif q_position in {26, 4, 28, 36}:
            return k_position == 30
        elif q_position in {25, 13, 35, 5}:
            return k_position == 33
        elif q_position in {6}:
            return k_position == 10
        elif q_position in {9, 10}:
            return k_position == 36
        elif q_position in {12, 22}:
            return k_position == 37
        elif q_position in {14}:
            return k_position == 13
        elif q_position in {17, 38, 15}:
            return k_position == 32
        elif q_position in {16, 19}:
            return k_position == 18
        elif q_position in {34, 20}:
            return k_position == 34
        elif q_position in {21}:
            return k_position == 6
        elif q_position in {23}:
            return k_position == 7
        elif q_position in {24}:
            return k_position == 11
        elif q_position in {32, 27}:
            return k_position == 28
        elif q_position in {37, 29, 39}:
            return k_position == 39
        elif q_position in {30}:
            return k_position == 29
        elif q_position in {33}:
            return k_position == 2

    num_attn_0_0_pattern = select(positions, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_token, k_token):
        if q_token in {"0", "2", "1", "3", "<s>", "</s>", "4"}:
            return k_token == ""

    num_attn_0_1_pattern = select(tokens, tokens, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(token, position):
        if token in {"0"}:
            return position == 36
        elif token in {"1"}:
            return position == 8
        elif token in {"2"}:
            return position == 38
        elif token in {"4", "3"}:
            return position == 31
        elif token in {"</s>"}:
            return position == 10
        elif token in {"<s>"}:
            return position == 11

    num_attn_0_2_pattern = select(positions, tokens, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(token, position):
        if token in {"0"}:
            return position == 30
        elif token in {"1"}:
            return position == 33
        elif token in {"2"}:
            return position == 32
        elif token in {"3"}:
            return position == 12
        elif token in {"4"}:
            return position == 36
        elif token in {"</s>"}:
            return position == 11
        elif token in {"<s>"}:
            return position == 10

    num_attn_0_3_pattern = select(positions, tokens, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(token, position):
        if token in {"0"}:
            return position == 32
        elif token in {"<s>", "1"}:
            return position == 10
        elif token in {"2"}:
            return position == 34
        elif token in {"3"}:
            return position == 33
        elif token in {"4"}:
            return position == 38
        elif token in {"</s>"}:
            return position == 11

    num_attn_0_4_pattern = select(positions, tokens, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(q_position, k_position):
        if q_position in {0, 37, 29, 30}:
            return k_position == 37
        elif q_position in {1, 18, 5, 14}:
            return k_position == 35
        elif q_position in {2, 38, 9, 12, 31}:
            return k_position == 38
        elif q_position in {3}:
            return k_position == 27
        elif q_position in {20, 27, 4, 13}:
            return k_position == 32
        elif q_position in {16, 10, 6}:
            return k_position == 34
        elif q_position in {26, 22, 7}:
            return k_position == 31
        elif q_position in {8}:
            return k_position == 13
        elif q_position in {25, 11}:
            return k_position == 39
        elif q_position in {33, 34, 15, 19, 24}:
            return k_position == 33
        elif q_position in {17}:
            return k_position == 21
        elif q_position in {21}:
            return k_position == 17
        elif q_position in {23}:
            return k_position == 3
        elif q_position in {28}:
            return k_position == 30
        elif q_position in {32}:
            return k_position == 26
        elif q_position in {35, 36}:
            return k_position == 36
        elif q_position in {39}:
            return k_position == 28

    num_attn_0_5_pattern = select(positions, positions, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(token, position):
        if token in {"0"}:
            return position == 8
        elif token in {"1"}:
            return position == 36
        elif token in {"2"}:
            return position == 37
        elif token in {"3"}:
            return position == 30
        elif token in {"4"}:
            return position == 31
        elif token in {"<s>", "</s>"}:
            return position == 13

    num_attn_0_6_pattern = select(positions, tokens, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(token, position):
        if token in {"0"}:
            return position == 35
        elif token in {"2", "1"}:
            return position == 30
        elif token in {"3"}:
            return position == 10
        elif token in {"4"}:
            return position == 37
        elif token in {"<s>", "</s>"}:
            return position == 12

    num_attn_0_7_pattern = select(positions, tokens, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_0_output, attn_0_5_output):
        key = (attn_0_0_output, attn_0_5_output)
        return 5

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_0_outputs, attn_0_5_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_4_output, attn_0_6_output):
        key = (attn_0_4_output, attn_0_6_output)
        return 4

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_4_outputs, attn_0_6_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_5_output, num_attn_0_2_output):
        key = (num_attn_0_5_output, num_attn_0_2_output)
        return 1

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_5_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(one, num_attn_0_3_output):
        key = (one, num_attn_0_3_output)
        return 5

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1) for k0, k1 in zip(ones, num_attn_0_3_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(attn_0_0_output, token):
        if attn_0_0_output in {"0", "4", "3"}:
            return token == ""
        elif attn_0_0_output in {"1"}:
            return token == "3"
        elif attn_0_0_output in {"2"}:
            return token == "</s>"
        elif attn_0_0_output in {"<s>", "</s>"}:
            return token == "<s>"

    attn_1_0_pattern = select_closest(tokens, attn_0_0_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_7_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_position, k_position):
        if q_position in {0, 19, 21, 23}:
            return k_position == 29
        elif q_position in {1}:
            return k_position == 5
        elif q_position in {32, 2}:
            return k_position == 16
        elif q_position in {3}:
            return k_position == 7
        elif q_position in {38, 18, 4, 6}:
            return k_position == 22
        elif q_position in {5, 9, 13, 15, 25, 28, 29}:
            return k_position == 1
        elif q_position in {7, 16, 17, 20, 24, 27}:
            return k_position == 2
        elif q_position in {8, 30}:
            return k_position == 11
        elif q_position in {10, 11, 31}:
            return k_position == 13
        elif q_position in {12}:
            return k_position == 15
        elif q_position in {14}:
            return k_position == 0
        elif q_position in {37, 22}:
            return k_position == 28
        elif q_position in {26}:
            return k_position == 3
        elif q_position in {33}:
            return k_position == 33
        elif q_position in {34}:
            return k_position == 26
        elif q_position in {35}:
            return k_position == 36
        elif q_position in {36}:
            return k_position == 35
        elif q_position in {39}:
            return k_position == 14

    attn_1_1_pattern = select_closest(positions, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, tokens)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(attn_0_0_output, token):
        if attn_0_0_output in {"0"}:
            return token == "</s>"
        elif attn_0_0_output in {"1"}:
            return token == "4"
        elif attn_0_0_output in {"2"}:
            return token == "3"
        elif attn_0_0_output in {"</s>", "3"}:
            return token == "1"
        elif attn_0_0_output in {"4"}:
            return token == "2"
        elif attn_0_0_output in {"<s>"}:
            return token == ""

    attn_1_2_pattern = select_closest(tokens, attn_0_0_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, tokens)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(attn_0_2_output, token):
        if attn_0_2_output in {"0", "1", "4"}:
            return token == ""
        elif attn_0_2_output in {"2"}:
            return token == "<pad>"
        elif attn_0_2_output in {"<s>", "3"}:
            return token == "4"
        elif attn_0_2_output in {"</s>"}:
            return token == "3"

    attn_1_3_pattern = select_closest(tokens, attn_0_2_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_1_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(attn_0_5_output, token):
        if attn_0_5_output in {"0"}:
            return token == "0"
        elif attn_0_5_output in {"</s>", "1"}:
            return token == ""
        elif attn_0_5_output in {"2"}:
            return token == "2"
        elif attn_0_5_output in {"3"}:
            return token == "3"
        elif attn_0_5_output in {"4"}:
            return token == "4"
        elif attn_0_5_output in {"<s>"}:
            return token == "1"

    attn_1_4_pattern = select_closest(tokens, attn_0_5_outputs, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, attn_0_1_outputs)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(q_position, k_position):
        if q_position in {0, 19}:
            return k_position == 7
        elif q_position in {1}:
            return k_position == 28
        elif q_position in {2, 29}:
            return k_position == 27
        elif q_position in {3}:
            return k_position == 18
        elif q_position in {9, 4, 12, 17}:
            return k_position == 10
        elif q_position in {16, 15, 5, 7}:
            return k_position == 13
        elif q_position in {37, 6}:
            return k_position == 19
        elif q_position in {8}:
            return k_position == 0
        elif q_position in {10, 20}:
            return k_position == 9
        elif q_position in {11, 13}:
            return k_position == 12
        elif q_position in {14, 22}:
            return k_position == 6
        elif q_position in {18}:
            return k_position == 11
        elif q_position in {21}:
            return k_position == 8
        elif q_position in {23}:
            return k_position == 29
        elif q_position in {24}:
            return k_position == 5
        elif q_position in {33, 34, 35, 38, 25, 31}:
            return k_position == 4
        elif q_position in {26}:
            return k_position == 3
        elif q_position in {27, 28}:
            return k_position == 1
        elif q_position in {30}:
            return k_position == 33
        elif q_position in {32}:
            return k_position == 17
        elif q_position in {36}:
            return k_position == 31
        elif q_position in {39}:
            return k_position == 26

    attn_1_5_pattern = select_closest(positions, positions, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, tokens)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(q_position, k_position):
        if q_position in {0, 28}:
            return k_position == 1
        elif q_position in {1}:
            return k_position == 26
        elif q_position in {2}:
            return k_position == 10
        elif q_position in {3}:
            return k_position == 16
        elif q_position in {4}:
            return k_position == 21
        elif q_position in {5}:
            return k_position == 20
        elif q_position in {6}:
            return k_position == 17
        elif q_position in {7}:
            return k_position == 22
        elif q_position in {8, 9, 10, 11, 12, 14, 15, 16}:
            return k_position == 0
        elif q_position in {32, 35, 36, 13}:
            return k_position == 15
        elif q_position in {17, 19, 21}:
            return k_position == 29
        elif q_position in {18}:
            return k_position == 5
        elif q_position in {20, 29}:
            return k_position == 28
        elif q_position in {24, 26, 27, 22}:
            return k_position == 2
        elif q_position in {25, 23}:
            return k_position == 3
        elif q_position in {33, 30}:
            return k_position == 19
        elif q_position in {38, 31}:
            return k_position == 14
        elif q_position in {34}:
            return k_position == 33
        elif q_position in {37}:
            return k_position == 4
        elif q_position in {39}:
            return k_position == 30

    attn_1_6_pattern = select_closest(positions, positions, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, tokens)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(attn_0_4_output, token):
        if attn_0_4_output in {"0"}:
            return token == "1"
        elif attn_0_4_output in {"1", "4", "3"}:
            return token == ""
        elif attn_0_4_output in {"<s>", "</s>", "2"}:
            return token == "0"

    attn_1_7_pattern = select_closest(tokens, attn_0_4_outputs, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, tokens)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(attn_0_1_output, attn_0_5_output):
        if attn_0_1_output in {"0", "2", "1", "3", "<s>", "</s>"}:
            return attn_0_5_output == ""
        elif attn_0_1_output in {"4"}:
            return attn_0_5_output == "4"

    num_attn_1_0_pattern = select(attn_0_5_outputs, attn_0_1_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_0_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(attn_0_1_output, attn_0_5_output):
        if attn_0_1_output in {"0", "1", "3", "<s>", "</s>", "4"}:
            return attn_0_5_output == ""
        elif attn_0_1_output in {"2"}:
            return attn_0_5_output == "2"

    num_attn_1_1_pattern = select(attn_0_5_outputs, attn_0_1_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_0_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(attn_0_6_output, token):
        if attn_0_6_output in {"0", "2", "3", "<s>", "</s>", "4"}:
            return token == ""
        elif attn_0_6_output in {"1"}:
            return token == "</s>"

    num_attn_1_2_pattern = select(tokens, attn_0_6_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_3_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(attn_0_7_output, attn_0_1_output):
        if attn_0_7_output in {"0", "2", "1", "3", "<s>", "</s>", "4"}:
            return attn_0_1_output == ""

    num_attn_1_3_pattern = select(attn_0_1_outputs, attn_0_7_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_1_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(attn_0_1_output, attn_0_0_output):
        if attn_0_1_output in {"0", "2", "1", "3", "<s>", "</s>", "4"}:
            return attn_0_0_output == ""

    num_attn_1_4_pattern = select(attn_0_0_outputs, attn_0_1_outputs, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_1_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(attn_0_0_output, attn_0_2_output):
        if attn_0_0_output in {"0", "1", "3", "<s>", "</s>", "4"}:
            return attn_0_2_output == ""
        elif attn_0_0_output in {"2"}:
            return attn_0_2_output == "2"

    num_attn_1_5_pattern = select(attn_0_2_outputs, attn_0_0_outputs, num_predicate_1_5)
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_4_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(attn_0_1_output, attn_0_5_output):
        if attn_0_1_output in {"0", "2", "3", "<s>", "</s>", "4"}:
            return attn_0_5_output == ""
        elif attn_0_1_output in {"1"}:
            return attn_0_5_output == "1"

    num_attn_1_6_pattern = select(attn_0_5_outputs, attn_0_1_outputs, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_0_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(attn_0_6_output, attn_0_0_output):
        if attn_0_6_output in {"0", "2", "3", "<s>", "</s>", "4"}:
            return attn_0_0_output == ""
        elif attn_0_6_output in {"1"}:
            return attn_0_0_output == "</s>"

    num_attn_1_7_pattern = select(attn_0_0_outputs, attn_0_6_outputs, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_2_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_7_output, attn_0_6_output):
        key = (attn_1_7_output, attn_0_6_output)
        return 38

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_7_outputs, attn_0_6_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(position, num_mlp_0_0_output):
        key = (position, num_mlp_0_0_output)
        if key in {
            (0, 12),
            (0, 19),
            (1, 12),
            (1, 19),
            (2, 19),
            (3, 0),
            (3, 1),
            (3, 3),
            (3, 4),
            (3, 5),
            (3, 7),
            (3, 8),
            (3, 9),
            (3, 10),
            (3, 11),
            (3, 12),
            (3, 13),
            (3, 14),
            (3, 15),
            (3, 16),
            (3, 17),
            (3, 18),
            (3, 19),
            (3, 20),
            (3, 21),
            (3, 22),
            (3, 23),
            (3, 24),
            (3, 25),
            (3, 26),
            (3, 27),
            (3, 28),
            (3, 29),
            (3, 30),
            (3, 31),
            (3, 32),
            (3, 33),
            (3, 34),
            (3, 35),
            (3, 36),
            (3, 37),
            (3, 38),
            (3, 39),
            (4, 12),
            (4, 19),
            (5, 12),
            (5, 19),
            (6, 12),
            (6, 19),
            (7, 12),
            (7, 19),
            (8, 12),
            (8, 19),
            (9, 19),
            (11, 12),
            (11, 19),
            (12, 12),
            (12, 14),
            (12, 19),
            (12, 39),
            (13, 12),
            (13, 14),
            (13, 19),
            (14, 12),
            (14, 14),
            (14, 19),
            (14, 20),
            (14, 21),
            (14, 30),
            (14, 39),
            (16, 12),
            (16, 19),
            (17, 12),
            (17, 14),
            (17, 19),
            (17, 20),
            (17, 30),
            (17, 39),
            (18, 12),
            (18, 19),
            (19, 0),
            (19, 1),
            (19, 2),
            (19, 3),
            (19, 4),
            (19, 5),
            (19, 6),
            (19, 7),
            (19, 8),
            (19, 9),
            (19, 10),
            (19, 11),
            (19, 12),
            (19, 13),
            (19, 14),
            (19, 15),
            (19, 16),
            (19, 17),
            (19, 18),
            (19, 19),
            (19, 20),
            (19, 21),
            (19, 22),
            (19, 23),
            (19, 24),
            (19, 25),
            (19, 26),
            (19, 27),
            (19, 28),
            (19, 29),
            (19, 30),
            (19, 31),
            (19, 32),
            (19, 33),
            (19, 34),
            (19, 35),
            (19, 36),
            (19, 37),
            (19, 38),
            (19, 39),
            (20, 0),
            (20, 1),
            (20, 3),
            (20, 4),
            (20, 5),
            (20, 7),
            (20, 8),
            (20, 9),
            (20, 10),
            (20, 11),
            (20, 12),
            (20, 13),
            (20, 14),
            (20, 15),
            (20, 16),
            (20, 17),
            (20, 18),
            (20, 19),
            (20, 20),
            (20, 21),
            (20, 22),
            (20, 23),
            (20, 24),
            (20, 25),
            (20, 26),
            (20, 27),
            (20, 28),
            (20, 29),
            (20, 30),
            (20, 31),
            (20, 32),
            (20, 33),
            (20, 34),
            (20, 35),
            (20, 36),
            (20, 37),
            (20, 38),
            (20, 39),
            (21, 12),
            (21, 19),
            (22, 12),
            (22, 19),
            (23, 12),
            (23, 19),
            (24, 0),
            (24, 1),
            (24, 3),
            (24, 4),
            (24, 5),
            (24, 7),
            (24, 8),
            (24, 9),
            (24, 10),
            (24, 11),
            (24, 12),
            (24, 13),
            (24, 14),
            (24, 15),
            (24, 16),
            (24, 17),
            (24, 18),
            (24, 19),
            (24, 20),
            (24, 21),
            (24, 22),
            (24, 23),
            (24, 24),
            (24, 25),
            (24, 26),
            (24, 27),
            (24, 28),
            (24, 29),
            (24, 30),
            (24, 31),
            (24, 32),
            (24, 33),
            (24, 34),
            (24, 35),
            (24, 36),
            (24, 37),
            (24, 38),
            (24, 39),
            (25, 12),
            (25, 19),
            (26, 12),
            (26, 19),
            (27, 12),
            (27, 19),
            (28, 12),
            (28, 19),
            (29, 12),
            (29, 19),
            (30, 12),
            (30, 19),
            (31, 12),
            (34, 12),
            (34, 19),
            (35, 12),
            (35, 14),
            (35, 19),
            (35, 20),
            (35, 39),
            (36, 12),
            (36, 19),
            (37, 12),
            (37, 19),
            (38, 12),
            (38, 19),
            (39, 12),
            (39, 14),
            (39, 19),
            (39, 20),
            (39, 39),
        }:
            return 17
        return 7

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(positions, num_mlp_0_0_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_4_output, num_attn_0_1_output):
        key = (num_attn_1_4_output, num_attn_0_1_output)
        return 15

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_4_outputs, num_attn_0_1_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_5_output, num_attn_1_3_output):
        key = (num_attn_1_5_output, num_attn_1_3_output)
        return 14

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_5_outputs, num_attn_1_3_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(attn_1_1_output, token):
        if attn_1_1_output in {"0", "2", "4", "</s>"}:
            return token == ""
        elif attn_1_1_output in {"1", "3"}:
            return token == "<s>"
        elif attn_1_1_output in {"<s>"}:
            return token == "</s>"

    attn_2_0_pattern = select_closest(tokens, attn_1_1_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_0_3_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(attn_0_1_output, token):
        if attn_0_1_output in {"<s>", "0"}:
            return token == ""
        elif attn_0_1_output in {"1"}:
            return token == "<s>"
        elif attn_0_1_output in {"2"}:
            return token == "4"
        elif attn_0_1_output in {"</s>", "3"}:
            return token == "2"
        elif attn_0_1_output in {"4"}:
            return token == "1"

    attn_2_1_pattern = select_closest(tokens, attn_0_1_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_2_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(attn_0_2_output, token):
        if attn_0_2_output in {"0"}:
            return token == "1"
        elif attn_0_2_output in {"1"}:
            return token == "<s>"
        elif attn_0_2_output in {"2", "4"}:
            return token == ""
        elif attn_0_2_output in {"3"}:
            return token == "</s>"
        elif attn_0_2_output in {"</s>"}:
            return token == "3"
        elif attn_0_2_output in {"<s>"}:
            return token == "0"

    attn_2_2_pattern = select_closest(tokens, attn_0_2_outputs, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_7_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(attn_1_1_output, token):
        if attn_1_1_output in {"0", "2", "1", "3"}:
            return token == "<s>"
        elif attn_1_1_output in {"<s>", "4"}:
            return token == ""
        elif attn_1_1_output in {"</s>"}:
            return token == "4"

    attn_2_3_pattern = select_closest(tokens, attn_1_1_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_5_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(attn_0_2_output, token):
        if attn_0_2_output in {"0", "2", "1", "<s>", "</s>", "4"}:
            return token == "3"
        elif attn_0_2_output in {"3"}:
            return token == "0"

    attn_2_4_pattern = select_closest(tokens, attn_0_2_outputs, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, tokens)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(attn_0_0_output, token):
        if attn_0_0_output in {"<s>", "0", "2", "1"}:
            return token == ""
        elif attn_0_0_output in {"4", "3"}:
            return token == "<s>"
        elif attn_0_0_output in {"</s>"}:
            return token == "4"

    attn_2_5_pattern = select_closest(tokens, attn_0_0_outputs, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_1_7_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(attn_0_0_output, token):
        if attn_0_0_output in {"0"}:
            return token == "0"
        elif attn_0_0_output in {"2", "1"}:
            return token == ""
        elif attn_0_0_output in {"3"}:
            return token == "2"
        elif attn_0_0_output in {"4"}:
            return token == "4"
        elif attn_0_0_output in {"</s>"}:
            return token == "</s>"
        elif attn_0_0_output in {"<s>"}:
            return token == "3"

    attn_2_6_pattern = select_closest(tokens, attn_0_0_outputs, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, attn_0_1_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(attn_0_2_output, token):
        if attn_0_2_output in {"0"}:
            return token == ""
        elif attn_0_2_output in {"1"}:
            return token == "<s>"
        elif attn_0_2_output in {"</s>", "2"}:
            return token == "0"
        elif attn_0_2_output in {"<s>", "3"}:
            return token == "4"
        elif attn_0_2_output in {"4"}:
            return token == "</s>"

    attn_2_7_pattern = select_closest(tokens, attn_0_2_outputs, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_1_0_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_0_1_output, attn_0_5_output):
        if attn_0_1_output in {"0", "2", "1", "<s>", "</s>", "4"}:
            return attn_0_5_output == ""
        elif attn_0_1_output in {"3"}:
            return attn_0_5_output == "3"

    num_attn_2_0_pattern = select(attn_0_5_outputs, attn_0_1_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_0_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_1_1_output, attn_1_0_output):
        if attn_1_1_output in {"0", "2", "1", "3", "<s>", "</s>"}:
            return attn_1_0_output == ""
        elif attn_1_1_output in {"4"}:
            return attn_1_0_output == "4"

    num_attn_2_1_pattern = select(attn_1_0_outputs, attn_1_1_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_5_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_0_6_output, attn_0_1_output):
        if attn_0_6_output in {"0"}:
            return attn_0_1_output == "</s>"
        elif attn_0_6_output in {"</s>", "2", "1", "4"}:
            return attn_0_1_output == ""
        elif attn_0_6_output in {"3"}:
            return attn_0_1_output == "<s>"
        elif attn_0_6_output in {"<s>"}:
            return attn_0_1_output == "0"

    num_attn_2_2_pattern = select(attn_0_1_outputs, attn_0_6_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_1_7_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_0_1_output, attn_0_5_output):
        if attn_0_1_output in {"0"}:
            return attn_0_5_output == "0"
        elif attn_0_1_output in {"2", "1", "3", "<s>", "</s>", "4"}:
            return attn_0_5_output == ""

    num_attn_2_3_pattern = select(attn_0_5_outputs, attn_0_1_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_0_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(attn_1_1_output, attn_0_4_output):
        if attn_1_1_output in {"0"}:
            return attn_0_4_output == "0"
        elif attn_1_1_output in {"2", "1", "3", "<s>", "</s>", "4"}:
            return attn_0_4_output == ""

    num_attn_2_4_pattern = select(attn_0_4_outputs, attn_1_1_outputs, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_0_5_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(attn_1_1_output, attn_0_2_output):
        if attn_1_1_output in {"0", "1", "3", "<s>", "</s>", "4"}:
            return attn_0_2_output == ""
        elif attn_1_1_output in {"2"}:
            return attn_0_2_output == "2"

    num_attn_2_5_pattern = select(attn_0_2_outputs, attn_1_1_outputs, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_0_6_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(attn_1_1_output, attn_0_0_output):
        if attn_1_1_output in {"0", "2", "3", "<s>", "</s>", "4"}:
            return attn_0_0_output == ""
        elif attn_1_1_output in {"1"}:
            return attn_0_0_output == "1"

    num_attn_2_6_pattern = select(attn_0_0_outputs, attn_1_1_outputs, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_0_0_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(attn_0_0_output, attn_0_4_output):
        if attn_0_0_output in {"0", "2", "1", "3", "<s>", "</s>", "4"}:
            return attn_0_4_output == ""

    num_attn_2_7_pattern = select(attn_0_4_outputs, attn_0_0_outputs, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_1_7_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_1_output, num_mlp_0_1_output):
        key = (attn_2_1_output, num_mlp_0_1_output)
        return 13

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_1_outputs, num_mlp_0_1_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_0_3_output, attn_2_7_output):
        key = (attn_0_3_output, attn_2_7_output)
        return 14

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_0_3_outputs, attn_2_7_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_2_4_output, num_attn_1_0_output):
        key = (num_attn_2_4_output, num_attn_1_0_output)
        return 34

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_2_4_outputs, num_attn_1_0_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_0_output, num_attn_2_3_output):
        key = (num_attn_2_0_output, num_attn_2_3_output)
        return 35

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_2_0_outputs, num_attn_2_3_outputs)
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


print(run(["<s>", "0", "3", "2", "3", "0", "2", "1", "3", "2", "</s>"]))
