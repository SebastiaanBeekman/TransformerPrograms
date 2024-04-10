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
        "output/length/rasp/reverse/trainlength20/s0/reverse_weights.csv",
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
        if q_position in {0, 8}:
            return k_position == 10
        elif q_position in {1}:
            return k_position == 8
        elif q_position in {2, 3}:
            return k_position == 12
        elif q_position in {4}:
            return k_position == 11
        elif q_position in {5}:
            return k_position == 19
        elif q_position in {11, 13, 6}:
            return k_position == 5
        elif q_position in {7}:
            return k_position == 4
        elif q_position in {9, 18, 14, 17}:
            return k_position == 1
        elif q_position in {10}:
            return k_position == 7
        elif q_position in {12}:
            return k_position == 6
        elif q_position in {16, 15}:
            return k_position == 2
        elif q_position in {19}:
            return k_position == 13
        elif q_position in {20, 29}:
            return k_position == 25
        elif q_position in {21}:
            return k_position == 27
        elif q_position in {26, 22}:
            return k_position == 23
        elif q_position in {25, 23}:
            return k_position == 28
        elif q_position in {24}:
            return k_position == 21
        elif q_position in {27, 28}:
            return k_position == 22

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 10
        elif q_position in {1, 2}:
            return k_position == 14
        elif q_position in {3}:
            return k_position == 15
        elif q_position in {4, 6}:
            return k_position == 12
        elif q_position in {19, 5}:
            return k_position == 13
        elif q_position in {10, 7}:
            return k_position == 0
        elif q_position in {8, 12, 13, 14}:
            return k_position == 4
        elif q_position in {9, 15}:
            return k_position == 3
        elif q_position in {11}:
            return k_position == 8
        elif q_position in {16}:
            return k_position == 2
        elif q_position in {17, 18}:
            return k_position == 1
        elif q_position in {20}:
            return k_position == 18
        elif q_position in {21, 22}:
            return k_position == 26
        elif q_position in {23}:
            return k_position == 24
        elif q_position in {24}:
            return k_position == 25
        elif q_position in {25}:
            return k_position == 23
        elif q_position in {26, 29}:
            return k_position == 21
        elif q_position in {27, 28}:
            return k_position == 29

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0, 16, 14}:
            return k_position == 19
        elif q_position in {1, 12, 6}:
            return k_position == 7
        elif q_position in {2, 11}:
            return k_position == 8
        elif q_position in {3}:
            return k_position == 3
        elif q_position in {8, 4}:
            return k_position == 11
        elif q_position in {5, 15}:
            return k_position == 4
        elif q_position in {19, 7}:
            return k_position == 5
        elif q_position in {9}:
            return k_position == 10
        elif q_position in {10}:
            return k_position == 9
        elif q_position in {13}:
            return k_position == 6
        elif q_position in {17}:
            return k_position == 2
        elif q_position in {18}:
            return k_position == 1
        elif q_position in {20}:
            return k_position == 22
        elif q_position in {26, 21}:
            return k_position == 24
        elif q_position in {24, 22}:
            return k_position == 26
        elif q_position in {23}:
            return k_position == 20
        elif q_position in {25}:
            return k_position == 29
        elif q_position in {27}:
            return k_position == 25
        elif q_position in {28}:
            return k_position == 23
        elif q_position in {29}:
            return k_position == 21

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 8, 6, 9}:
            return k_position == 19
        elif q_position in {1}:
            return k_position == 18
        elif q_position in {2}:
            return k_position == 17
        elif q_position in {3}:
            return k_position == 16
        elif q_position in {4}:
            return k_position == 15
        elif q_position in {5}:
            return k_position == 14
        elif q_position in {7}:
            return k_position == 12
        elif q_position in {10}:
            return k_position == 2
        elif q_position in {11, 15, 16, 17, 18}:
            return k_position == 1
        elif q_position in {12}:
            return k_position == 7
        elif q_position in {13}:
            return k_position == 6
        elif q_position in {14}:
            return k_position == 3
        elif q_position in {19}:
            return k_position == 5
        elif q_position in {27, 20, 28, 23}:
            return k_position == 26
        elif q_position in {21}:
            return k_position == 28
        elif q_position in {22}:
            return k_position == 29
        elif q_position in {24}:
            return k_position == 27
        elif q_position in {25}:
            return k_position == 21
        elif q_position in {26, 29}:
            return k_position == 25

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(q_position, k_position):
        if q_position in {0}:
            return k_position == 10
        elif q_position in {1}:
            return k_position == 15
        elif q_position in {9, 2}:
            return k_position == 12
        elif q_position in {3, 4}:
            return k_position == 14
        elif q_position in {5, 22}:
            return k_position == 19
        elif q_position in {6}:
            return k_position == 9
        elif q_position in {7, 8, 10, 11, 13, 15, 16, 17, 18}:
            return k_position == 1
        elif q_position in {12, 14}:
            return k_position == 3
        elif q_position in {19}:
            return k_position == 5
        elif q_position in {26, 20}:
            return k_position == 26
        elif q_position in {21, 23}:
            return k_position == 27
        elif q_position in {24, 27}:
            return k_position == 21
        elif q_position in {25}:
            return k_position == 23
        elif q_position in {28}:
            return k_position == 25
        elif q_position in {29}:
            return k_position == 24

    attn_0_4_pattern = select_closest(positions, positions, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(q_position, k_position):
        if q_position in {0}:
            return k_position == 10
        elif q_position in {1}:
            return k_position == 16
        elif q_position in {2}:
            return k_position == 15
        elif q_position in {3, 6}:
            return k_position == 11
        elif q_position in {4, 7}:
            return k_position == 8
        elif q_position in {5, 11, 12, 13, 14, 15, 16}:
            return k_position == 2
        elif q_position in {8}:
            return k_position == 19
        elif q_position in {9, 10}:
            return k_position == 4
        elif q_position in {17, 18}:
            return k_position == 1
        elif q_position in {19}:
            return k_position == 13
        elif q_position in {25, 29, 20, 21}:
            return k_position == 25
        elif q_position in {22}:
            return k_position == 20
        elif q_position in {24, 23}:
            return k_position == 23
        elif q_position in {26, 28}:
            return k_position == 27
        elif q_position in {27}:
            return k_position == 24

    attn_0_5_pattern = select_closest(positions, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(q_position, k_position):
        if q_position in {0}:
            return k_position == 10
        elif q_position in {1, 19, 4}:
            return k_position == 13
        elif q_position in {2, 3}:
            return k_position == 7
        elif q_position in {5, 13, 15, 16, 18}:
            return k_position == 1
        elif q_position in {6}:
            return k_position == 2
        elif q_position in {7}:
            return k_position == 3
        elif q_position in {8, 9}:
            return k_position == 5
        elif q_position in {10}:
            return k_position == 19
        elif q_position in {11}:
            return k_position == 6
        elif q_position in {17, 27, 12}:
            return k_position == 17
        elif q_position in {14}:
            return k_position == 16
        elif q_position in {20, 21}:
            return k_position == 28
        elif q_position in {22}:
            return k_position == 0
        elif q_position in {26, 23}:
            return k_position == 24
        elif q_position in {24}:
            return k_position == 20
        elif q_position in {25}:
            return k_position == 29
        elif q_position in {28, 29}:
            return k_position == 23

    attn_0_6_pattern = select_closest(positions, positions, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(q_position, k_position):
        if q_position in {0, 9, 10, 15, 17, 21}:
            return k_position == 19
        elif q_position in {1}:
            return k_position == 8
        elif q_position in {2, 3}:
            return k_position == 6
        elif q_position in {4}:
            return k_position == 7
        elif q_position in {5}:
            return k_position == 9
        elif q_position in {6}:
            return k_position == 13
        elif q_position in {7}:
            return k_position == 12
        elif q_position in {8, 16}:
            return k_position == 3
        elif q_position in {11}:
            return k_position == 0
        elif q_position in {12}:
            return k_position == 2
        elif q_position in {13}:
            return k_position == 4
        elif q_position in {19, 14}:
            return k_position == 5
        elif q_position in {18}:
            return k_position == 1
        elif q_position in {20, 23}:
            return k_position == 29
        elif q_position in {25, 22}:
            return k_position == 20
        elif q_position in {24}:
            return k_position == 22
        elif q_position in {26, 28}:
            return k_position == 24
        elif q_position in {27}:
            return k_position == 23
        elif q_position in {29}:
            return k_position == 25

    attn_0_7_pattern = select_closest(positions, positions, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(token, position):
        if token in {"3", "0"}:
            return position == 21
        elif token in {"4", "1", "2"}:
            return position == 19
        elif token in {"<s>", "</s>"}:
            return position == 9

    num_attn_0_0_pattern = select(positions, tokens, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_position, k_position):
        if q_position in {0, 14, 23}:
            return k_position == 23
        elif q_position in {1, 27, 6, 7}:
            return k_position == 9
        elif q_position in {8, 9, 2, 10}:
            return k_position == 12
        elif q_position in {17, 18, 3, 4}:
            return k_position == 19
        elif q_position in {21, 5}:
            return k_position == 26
        elif q_position in {24, 11, 20}:
            return k_position == 21
        elif q_position in {26, 12, 13}:
            return k_position == 20
        elif q_position in {15}:
            return k_position == 2
        elif q_position in {16}:
            return k_position == 25
        elif q_position in {19, 29}:
            return k_position == 24
        elif q_position in {22}:
            return k_position == 29
        elif q_position in {25, 28}:
            return k_position == 28

    num_attn_0_1_pattern = select(positions, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(token, position):
        if token in {"0"}:
            return position == 28
        elif token in {"1"}:
            return position == 24
        elif token in {"2"}:
            return position == 21
        elif token in {"3"}:
            return position == 8
        elif token in {"4"}:
            return position == 20
        elif token in {"</s>"}:
            return position == 19
        elif token in {"<s>"}:
            return position == 23

    num_attn_0_2_pattern = select(positions, tokens, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(q_position, k_position):
        if q_position in {0, 18}:
            return k_position == 23
        elif q_position in {1, 20, 28}:
            return k_position == 26
        elif q_position in {2, 19}:
            return k_position == 25
        elif q_position in {3, 9, 12, 27, 29}:
            return k_position == 24
        elif q_position in {4}:
            return k_position == 20
        elif q_position in {5}:
            return k_position == 28
        elif q_position in {16, 17, 6, 14}:
            return k_position == 19
        elif q_position in {7}:
            return k_position == 11
        elif q_position in {8, 13, 23}:
            return k_position == 21
        elif q_position in {10}:
            return k_position == 22
        elif q_position in {26, 11}:
            return k_position == 27
        elif q_position in {15}:
            return k_position == 18
        elif q_position in {21}:
            return k_position == 9
        elif q_position in {24, 25, 22}:
            return k_position == 3

    num_attn_0_3_pattern = select(positions, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(token, position):
        if token in {"0"}:
            return position == 24
        elif token in {"1"}:
            return position == 20
        elif token in {"<s>", "4", "2"}:
            return position == 8
        elif token in {"3"}:
            return position == 25
        elif token in {"</s>"}:
            return position == 6

    num_attn_0_4_pattern = select(positions, tokens, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(token, position):
        if token in {"0"}:
            return position == 28
        elif token in {"1"}:
            return position == 22
        elif token in {"2"}:
            return position == 8
        elif token in {"3"}:
            return position == 26
        elif token in {"4"}:
            return position == 25
        elif token in {"</s>"}:
            return position == 9
        elif token in {"<s>"}:
            return position == 20

    num_attn_0_5_pattern = select(positions, tokens, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(token, position):
        if token in {"3", "0"}:
            return position == 19
        elif token in {"1"}:
            return position == 26
        elif token in {"<s>", "2"}:
            return position == 27
        elif token in {"4"}:
            return position == 21
        elif token in {"</s>"}:
            return position == 10

    num_attn_0_6_pattern = select(positions, tokens, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(token, position):
        if token in {"0"}:
            return position == 8
        elif token in {"3", "4", "1", "2"}:
            return position == 19
        elif token in {"<s>", "</s>"}:
            return position == 9

    num_attn_0_7_pattern = select(positions, tokens, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_7_output, attn_0_2_output):
        key = (attn_0_7_output, attn_0_2_output)
        if key in {
            ("0", "0"),
            ("0", "1"),
            ("0", "2"),
            ("0", "3"),
            ("0", "4"),
            ("0", "<s>"),
            ("1", "0"),
            ("2", "0"),
            ("3", "0"),
            ("4", "0"),
            ("<s>", "0"),
        }:
            return 13
        elif key in {("0", "</s>"), ("</s>", "0")}:
            return 4
        elif key in {
            ("1", "</s>"),
            ("2", "</s>"),
            ("3", "</s>"),
            ("</s>", "1"),
            ("</s>", "2"),
            ("</s>", "3"),
            ("</s>", "</s>"),
            ("</s>", "<s>"),
            ("<s>", "</s>"),
        }:
            return 26
        elif key in {("4", "</s>"), ("</s>", "4")}:
            return 7
        return 6

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_7_outputs, attn_0_2_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_2_output, attn_0_7_output):
        key = (attn_0_2_output, attn_0_7_output)
        if key in {
            ("0", "</s>"),
            ("1", "</s>"),
            ("2", "</s>"),
            ("4", "</s>"),
            ("</s>", "0"),
            ("</s>", "1"),
            ("</s>", "</s>"),
            ("</s>", "<s>"),
            ("<s>", "</s>"),
        }:
            return 11
        elif key in {
            ("0", "3"),
            ("1", "3"),
            ("2", "3"),
            ("3", "3"),
            ("3", "</s>"),
            ("4", "3"),
            ("</s>", "2"),
            ("</s>", "3"),
            ("</s>", "4"),
            ("<s>", "3"),
        }:
            return 13
        return 10

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_0_7_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_5_output, num_attn_0_6_output):
        key = (num_attn_0_5_output, num_attn_0_6_output)
        return 9

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_5_outputs, num_attn_0_6_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_1_output, num_attn_0_2_output):
        key = (num_attn_0_1_output, num_attn_0_2_output)
        return 5

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(attn_0_3_output, token):
        if attn_0_3_output in {"0"}:
            return token == "0"
        elif attn_0_3_output in {"1"}:
            return token == "1"
        elif attn_0_3_output in {"2"}:
            return token == "2"
        elif attn_0_3_output in {"<s>", "3"}:
            return token == "3"
        elif attn_0_3_output in {"4"}:
            return token == "4"
        elif attn_0_3_output in {"</s>"}:
            return token == ""

    attn_1_0_pattern = select_closest(tokens, attn_0_3_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_4_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(attn_0_4_output, token):
        if attn_0_4_output in {"2", "0"}:
            return token == "4"
        elif attn_0_4_output in {"4", "1"}:
            return token == "0"
        elif attn_0_4_output in {"</s>", "3"}:
            return token == ""
        elif attn_0_4_output in {"<s>"}:
            return token == "3"

    attn_1_1_pattern = select_closest(tokens, attn_0_4_outputs, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, tokens)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(position, token):
        if position in {0}:
            return token == "1"
        elif position in {1, 13, 16, 19, 20, 23, 25, 29}:
            return token == ""
        elif position in {2, 5, 8, 10, 11, 14}:
            return token == "3"
        elif position in {3, 4, 6, 7, 9, 12, 21, 22, 24, 26, 27, 28}:
            return token == "0"
        elif position in {17, 15}:
            return token == "<s>"
        elif position in {18}:
            return token == "</s>"

    attn_1_2_pattern = select_closest(tokens, positions, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_4_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(attn_0_4_output, token):
        if attn_0_4_output in {"<s>", "1", "0"}:
            return token == ""
        elif attn_0_4_output in {"4", "2", "</s>", "3"}:
            return token == "2"

    attn_1_3_pattern = select_closest(tokens, attn_0_4_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, tokens)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(q_position, k_position):
        if q_position in {0, 8}:
            return k_position == 6
        elif q_position in {1}:
            return k_position == 12
        elif q_position in {2}:
            return k_position == 14
        elif q_position in {3}:
            return k_position == 8
        elif q_position in {4}:
            return k_position == 9
        elif q_position in {5, 7}:
            return k_position == 11
        elif q_position in {6}:
            return k_position == 19
        elif q_position in {9, 26}:
            return k_position == 10
        elif q_position in {10, 11, 13}:
            return k_position == 7
        elif q_position in {12}:
            return k_position == 13
        elif q_position in {14}:
            return k_position == 15
        elif q_position in {16, 15}:
            return k_position == 16
        elif q_position in {17, 18}:
            return k_position == 18
        elif q_position in {19}:
            return k_position == 0
        elif q_position in {20, 22}:
            return k_position == 21
        elif q_position in {21}:
            return k_position == 20
        elif q_position in {23}:
            return k_position == 17
        elif q_position in {24, 27}:
            return k_position == 25
        elif q_position in {25}:
            return k_position == 22
        elif q_position in {28}:
            return k_position == 23
        elif q_position in {29}:
            return k_position == 24

    attn_1_4_pattern = select_closest(positions, positions, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, attn_0_6_outputs)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(attn_0_7_output, token):
        if attn_0_7_output in {"2", "0"}:
            return token == ""
        elif attn_0_7_output in {"</s>", "<s>", "1", "3", "4"}:
            return token == "0"

    attn_1_5_pattern = select_closest(tokens, attn_0_7_outputs, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, tokens)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(q_position, k_position):
        if q_position in {0, 22, 25, 26, 29}:
            return k_position == 1
        elif q_position in {1, 2, 4, 5, 16}:
            return k_position == 17
        elif q_position in {3, 13}:
            return k_position == 14
        elif q_position in {6}:
            return k_position == 19
        elif q_position in {7}:
            return k_position == 8
        elif q_position in {8, 9}:
            return k_position == 10
        elif q_position in {10}:
            return k_position == 12
        elif q_position in {11}:
            return k_position == 7
        elif q_position in {12}:
            return k_position == 11
        elif q_position in {14}:
            return k_position == 15
        elif q_position in {17, 19, 15}:
            return k_position == 16
        elif q_position in {18}:
            return k_position == 18
        elif q_position in {20, 23}:
            return k_position == 21
        elif q_position in {21}:
            return k_position == 2
        elif q_position in {24, 28}:
            return k_position == 25
        elif q_position in {27}:
            return k_position == 27

    attn_1_6_pattern = select_closest(positions, positions, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, attn_0_0_outputs)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(attn_0_7_output, token):
        if attn_0_7_output in {"2", "1", "0"}:
            return token == "3"
        elif attn_0_7_output in {"<s>", "4", "3"}:
            return token == "2"
        elif attn_0_7_output in {"</s>"}:
            return token == "</s>"

    attn_1_7_pattern = select_closest(tokens, attn_0_7_outputs, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, attn_0_4_outputs)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(attn_0_4_output, attn_0_0_output):
        if attn_0_4_output in {"2", "</s>", "<s>", "0", "3", "4"}:
            return attn_0_0_output == ""
        elif attn_0_4_output in {"1"}:
            return attn_0_0_output == "1"

    num_attn_1_0_pattern = select(attn_0_0_outputs, attn_0_4_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_1_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(q_num_mlp_0_0_output, k_num_mlp_0_0_output):
        if q_num_mlp_0_0_output in {0, 29}:
            return k_num_mlp_0_0_output == 14
        elif q_num_mlp_0_0_output in {1}:
            return k_num_mlp_0_0_output == 7
        elif q_num_mlp_0_0_output in {2, 13}:
            return k_num_mlp_0_0_output == 28
        elif q_num_mlp_0_0_output in {3}:
            return k_num_mlp_0_0_output == 27
        elif q_num_mlp_0_0_output in {28, 4, 21}:
            return k_num_mlp_0_0_output == 18
        elif q_num_mlp_0_0_output in {27, 5}:
            return k_num_mlp_0_0_output == 19
        elif q_num_mlp_0_0_output in {24, 10, 6, 23}:
            return k_num_mlp_0_0_output == 26
        elif q_num_mlp_0_0_output in {7}:
            return k_num_mlp_0_0_output == 6
        elif q_num_mlp_0_0_output in {8}:
            return k_num_mlp_0_0_output == 8
        elif q_num_mlp_0_0_output in {9}:
            return k_num_mlp_0_0_output == 13
        elif q_num_mlp_0_0_output in {11, 22}:
            return k_num_mlp_0_0_output == 4
        elif q_num_mlp_0_0_output in {12}:
            return k_num_mlp_0_0_output == 17
        elif q_num_mlp_0_0_output in {14}:
            return k_num_mlp_0_0_output == 25
        elif q_num_mlp_0_0_output in {15}:
            return k_num_mlp_0_0_output == 1
        elif q_num_mlp_0_0_output in {16, 17}:
            return k_num_mlp_0_0_output == 12
        elif q_num_mlp_0_0_output in {25, 18}:
            return k_num_mlp_0_0_output == 11
        elif q_num_mlp_0_0_output in {19}:
            return k_num_mlp_0_0_output == 2
        elif q_num_mlp_0_0_output in {26, 20}:
            return k_num_mlp_0_0_output == 10

    num_attn_1_1_pattern = select(
        num_mlp_0_0_outputs, num_mlp_0_0_outputs, num_predicate_1_1
    )
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_1_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(attn_0_3_output, attn_0_7_output):
        if attn_0_3_output in {"2", "</s>", "<s>", "0", "3"}:
            return attn_0_7_output == ""
        elif attn_0_3_output in {"1"}:
            return attn_0_7_output == "<pad>"
        elif attn_0_3_output in {"4"}:
            return attn_0_7_output == "</s>"

    num_attn_1_2_pattern = select(attn_0_7_outputs, attn_0_3_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_3_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(attn_0_4_output, attn_0_0_output):
        if attn_0_4_output in {"2", "</s>", "<s>", "1", "0"}:
            return attn_0_0_output == ""
        elif attn_0_4_output in {"3"}:
            return attn_0_0_output == "<pad>"
        elif attn_0_4_output in {"4"}:
            return attn_0_0_output == "4"

    num_attn_1_3_pattern = select(attn_0_0_outputs, attn_0_4_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_1_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(attn_0_5_output, attn_0_4_output):
        if attn_0_5_output in {"4", "2", "</s>", "0"}:
            return attn_0_4_output == ""
        elif attn_0_5_output in {"1", "3"}:
            return attn_0_4_output == "<s>"
        elif attn_0_5_output in {"<s>"}:
            return attn_0_4_output == "<pad>"

    num_attn_1_4_pattern = select(attn_0_4_outputs, attn_0_5_outputs, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_4_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(attn_0_2_output, mlp_0_0_output):
        if attn_0_2_output in {"1", "</s>", "0"}:
            return mlp_0_0_output == 19
        elif attn_0_2_output in {"2"}:
            return mlp_0_0_output == 26
        elif attn_0_2_output in {"3"}:
            return mlp_0_0_output == 29
        elif attn_0_2_output in {"<s>", "4"}:
            return mlp_0_0_output == 4

    num_attn_1_5_pattern = select(mlp_0_0_outputs, attn_0_2_outputs, num_predicate_1_5)
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_3_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(attn_0_2_output, mlp_0_1_output):
        if attn_0_2_output in {"0"}:
            return mlp_0_1_output == 14
        elif attn_0_2_output in {"1"}:
            return mlp_0_1_output == 23
        elif attn_0_2_output in {"2"}:
            return mlp_0_1_output == 8
        elif attn_0_2_output in {"3"}:
            return mlp_0_1_output == 11
        elif attn_0_2_output in {"4"}:
            return mlp_0_1_output == 28
        elif attn_0_2_output in {"</s>"}:
            return mlp_0_1_output == 27
        elif attn_0_2_output in {"<s>"}:
            return mlp_0_1_output == 17

    num_attn_1_6_pattern = select(mlp_0_1_outputs, attn_0_2_outputs, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_3_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(position, attn_0_2_output):
        if position in {0, 3, 29}:
            return attn_0_2_output == "</s>"
        elif position in {1}:
            return attn_0_2_output == "<s>"
        elif position in {
            2,
            4,
            5,
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
        }:
            return attn_0_2_output == ""
        elif position in {24, 6}:
            return attn_0_2_output == "<pad>"

    num_attn_1_7_pattern = select(attn_0_2_outputs, positions, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_2_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(num_mlp_0_1_output, attn_1_1_output):
        key = (num_mlp_0_1_output, attn_1_1_output)
        return 0

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(num_mlp_0_1_outputs, attn_1_1_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_0_2_output, attn_0_7_output):
        key = (attn_0_2_output, attn_0_7_output)
        if key in {
            ("0", "</s>"),
            ("1", "</s>"),
            ("3", "</s>"),
            ("4", "</s>"),
            ("</s>", "0"),
            ("</s>", "1"),
            ("</s>", "3"),
            ("</s>", "4"),
            ("</s>", "</s>"),
            ("</s>", "<s>"),
            ("<s>", "</s>"),
        }:
            return 1
        elif key in {
            ("0", "2"),
            ("1", "2"),
            ("2", "2"),
            ("2", "<s>"),
            ("3", "2"),
            ("4", "2"),
            ("<s>", "2"),
        }:
            return 27
        return 13

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_0_7_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_7_output, num_attn_1_6_output):
        key = (num_attn_1_7_output, num_attn_1_6_output)
        return 8

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_7_outputs, num_attn_1_6_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_5_output, num_attn_1_1_output):
        key = (num_attn_1_5_output, num_attn_1_1_output)
        return 4

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_5_outputs, num_attn_1_1_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(attn_0_4_output, token):
        if attn_0_4_output in {"</s>", "0"}:
            return token == ""
        elif attn_0_4_output in {"1"}:
            return token == "<s>"
        elif attn_0_4_output in {"2"}:
            return token == "1"
        elif attn_0_4_output in {"3"}:
            return token == "3"
        elif attn_0_4_output in {"4"}:
            return token == "4"
        elif attn_0_4_output in {"<s>"}:
            return token == "0"

    attn_2_0_pattern = select_closest(tokens, attn_0_4_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_6_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(attn_0_1_output, attn_1_1_output):
        if attn_0_1_output in {"0"}:
            return attn_1_1_output == "</s>"
        elif attn_0_1_output in {"4", "1"}:
            return attn_1_1_output == ""
        elif attn_0_1_output in {"2", "3"}:
            return attn_1_1_output == "<s>"
        elif attn_0_1_output in {"<s>", "</s>"}:
            return attn_1_1_output == "0"

    attn_2_1_pattern = select_closest(attn_1_1_outputs, attn_0_1_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_1_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(attn_0_5_output, token):
        if attn_0_5_output in {"2", "<s>", "1", "0", "3"}:
            return token == "<s>"
        elif attn_0_5_output in {"4", "</s>"}:
            return token == ""

    attn_2_2_pattern = select_closest(tokens, attn_0_5_outputs, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_5_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(attn_0_4_output, token):
        if attn_0_4_output in {"0"}:
            return token == "0"
        elif attn_0_4_output in {"1"}:
            return token == "1"
        elif attn_0_4_output in {"2"}:
            return token == "<s>"
        elif attn_0_4_output in {"<s>", "3"}:
            return token == "3"
        elif attn_0_4_output in {"4"}:
            return token == "4"
        elif attn_0_4_output in {"</s>"}:
            return token == ""

    attn_2_3_pattern = select_closest(tokens, attn_0_4_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_6_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(attn_0_3_output, token):
        if attn_0_3_output in {"0"}:
            return token == "<s>"
        elif attn_0_3_output in {"<s>", "1"}:
            return token == "1"
        elif attn_0_3_output in {"2"}:
            return token == "2"
        elif attn_0_3_output in {"3"}:
            return token == "3"
        elif attn_0_3_output in {"4", "</s>"}:
            return token == ""

    attn_2_4_pattern = select_closest(tokens, attn_0_3_outputs, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_1_6_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(attn_0_4_output, position):
        if attn_0_4_output in {"0"}:
            return position == 16
        elif attn_0_4_output in {"4", "1", "3"}:
            return position == 17
        elif attn_0_4_output in {"2"}:
            return position == 4
        elif attn_0_4_output in {"</s>"}:
            return position == 12
        elif attn_0_4_output in {"<s>"}:
            return position == 13

    attn_2_5_pattern = select_closest(positions, attn_0_4_outputs, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_1_6_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(attn_0_4_output, token):
        if attn_0_4_output in {"2", "</s>", "<s>", "1", "0", "4"}:
            return token == "4"
        elif attn_0_4_output in {"3"}:
            return token == ""

    attn_2_6_pattern = select_closest(tokens, attn_0_4_outputs, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, tokens)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(attn_0_4_output, position):
        if attn_0_4_output in {"4", "0"}:
            return position == 16
        elif attn_0_4_output in {"1"}:
            return position == 17
        elif attn_0_4_output in {"2", "3"}:
            return position == 15
        elif attn_0_4_output in {"</s>"}:
            return position == 11
        elif attn_0_4_output in {"<s>"}:
            return position == 13

    attn_2_7_pattern = select_closest(positions, attn_0_4_outputs, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_0_4_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_0_4_output, attn_0_0_output):
        if attn_0_4_output in {"</s>", "<s>", "0", "3", "4"}:
            return attn_0_0_output == ""
        elif attn_0_4_output in {"1"}:
            return attn_0_0_output == "<pad>"
        elif attn_0_4_output in {"2"}:
            return attn_0_0_output == "2"

    num_attn_2_0_pattern = select(attn_0_0_outputs, attn_0_4_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_1_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_0_4_output, attn_0_0_output):
        if attn_0_4_output in {"0"}:
            return attn_0_0_output == "0"
        elif attn_0_4_output in {"2", "</s>", "<s>", "1", "3", "4"}:
            return attn_0_0_output == ""

    num_attn_2_1_pattern = select(attn_0_0_outputs, attn_0_4_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_1_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_0_6_output, attn_1_1_output):
        if attn_0_6_output in {"2", "</s>", "<s>", "1", "0", "3"}:
            return attn_1_1_output == ""
        elif attn_0_6_output in {"4"}:
            return attn_1_1_output == "0"

    num_attn_2_2_pattern = select(attn_1_1_outputs, attn_0_6_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_1_2_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_1_4_output, attn_0_3_output):
        if attn_1_4_output in {"</s>", "<s>", "1", "0", "3", "4"}:
            return attn_0_3_output == ""
        elif attn_1_4_output in {"2"}:
            return attn_0_3_output == "2"

    num_attn_2_3_pattern = select(attn_0_3_outputs, attn_1_4_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_1_3_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(attn_0_4_output, attn_1_4_output):
        if attn_0_4_output in {"2", "</s>", "<s>", "0", "3", "4"}:
            return attn_1_4_output == ""
        elif attn_0_4_output in {"1"}:
            return attn_1_4_output == "1"

    num_attn_2_4_pattern = select(attn_1_4_outputs, attn_0_4_outputs, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_1_1_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(q_attn_1_4_output, k_attn_1_4_output):
        if q_attn_1_4_output in {"</s>", "<s>", "1", "0", "3"}:
            return k_attn_1_4_output == ""
        elif q_attn_1_4_output in {"4", "2"}:
            return k_attn_1_4_output == "<pad>"

    num_attn_2_5_pattern = select(attn_1_4_outputs, attn_1_4_outputs, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_0_1_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(attn_0_4_output, attn_1_4_output):
        if attn_0_4_output in {"<s>", "2", "</s>", "0"}:
            return attn_1_4_output == ""
        elif attn_0_4_output in {"4", "1"}:
            return attn_1_4_output == "<pad>"
        elif attn_0_4_output in {"3"}:
            return attn_1_4_output == "3"

    num_attn_2_6_pattern = select(attn_1_4_outputs, attn_0_4_outputs, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_1_1_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(attn_0_4_output, attn_0_6_output):
        if attn_0_4_output in {"3", "2", "1", "0"}:
            return attn_0_6_output == ""
        elif attn_0_4_output in {"<s>", "4"}:
            return attn_0_6_output == "4"
        elif attn_0_4_output in {"</s>"}:
            return attn_0_6_output == "<pad>"

    num_attn_2_7_pattern = select(attn_0_6_outputs, attn_0_4_outputs, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_0_1_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_0_2_output, attn_0_7_output):
        key = (attn_0_2_output, attn_0_7_output)
        if key in {
            ("0", "</s>"),
            ("2", "</s>"),
            ("3", "</s>"),
            ("4", "</s>"),
            ("</s>", "0"),
            ("</s>", "2"),
            ("</s>", "3"),
            ("</s>", "4"),
            ("</s>", "</s>"),
            ("</s>", "<s>"),
            ("<s>", "</s>"),
        }:
            return 6
        elif key in {
            ("0", "1"),
            ("1", "1"),
            ("1", "</s>"),
            ("2", "1"),
            ("3", "1"),
            ("4", "1"),
            ("</s>", "1"),
            ("<s>", "1"),
        }:
            return 10
        return 17

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_0_7_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_2_0_output, num_mlp_1_0_output):
        key = (attn_2_0_output, num_mlp_1_0_output)
        return 6

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_2_0_outputs, num_mlp_1_0_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_6_output, num_attn_1_0_output):
        key = (num_attn_1_6_output, num_attn_1_0_output)
        return 22

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_1_6_outputs, num_attn_1_0_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_0_output, num_attn_0_1_output):
        key = (num_attn_2_0_output, num_attn_0_1_output)
        if key in {
            (15, 0),
            (16, 0),
            (17, 0),
            (17, 1),
            (18, 0),
            (18, 1),
            (19, 0),
            (19, 1),
            (20, 0),
            (20, 1),
            (20, 2),
            (21, 0),
            (21, 1),
            (21, 2),
            (22, 0),
            (22, 1),
            (22, 2),
            (22, 3),
            (23, 0),
            (23, 1),
            (23, 2),
            (23, 3),
            (24, 0),
            (24, 1),
            (24, 2),
            (24, 3),
            (24, 4),
            (25, 0),
            (25, 1),
            (25, 2),
            (25, 3),
            (25, 4),
            (26, 0),
            (26, 1),
            (26, 2),
            (26, 3),
            (26, 4),
            (27, 0),
            (27, 1),
            (27, 2),
            (27, 3),
            (27, 4),
            (27, 5),
            (28, 0),
            (28, 1),
            (28, 2),
            (28, 3),
            (28, 4),
            (28, 5),
            (29, 0),
            (29, 1),
            (29, 2),
            (29, 3),
            (29, 4),
            (29, 5),
            (29, 6),
            (30, 0),
            (30, 1),
            (30, 2),
            (30, 3),
            (30, 4),
            (30, 5),
            (30, 6),
            (31, 0),
            (31, 1),
            (31, 2),
            (31, 3),
            (31, 4),
            (31, 5),
            (31, 6),
            (31, 7),
            (32, 0),
            (32, 1),
            (32, 2),
            (32, 3),
            (32, 4),
            (32, 5),
            (32, 6),
            (32, 7),
            (33, 0),
            (33, 1),
            (33, 2),
            (33, 3),
            (33, 4),
            (33, 5),
            (33, 6),
            (33, 7),
            (33, 8),
            (34, 0),
            (34, 1),
            (34, 2),
            (34, 3),
            (34, 4),
            (34, 5),
            (34, 6),
            (34, 7),
            (34, 8),
            (35, 0),
            (35, 1),
            (35, 2),
            (35, 3),
            (35, 4),
            (35, 5),
            (35, 6),
            (35, 7),
            (35, 8),
            (36, 0),
            (36, 1),
            (36, 2),
            (36, 3),
            (36, 4),
            (36, 5),
            (36, 6),
            (36, 7),
            (36, 8),
            (36, 9),
            (37, 0),
            (37, 1),
            (37, 2),
            (37, 3),
            (37, 4),
            (37, 5),
            (37, 6),
            (37, 7),
            (37, 8),
            (37, 9),
            (38, 0),
            (38, 1),
            (38, 2),
            (38, 3),
            (38, 4),
            (38, 5),
            (38, 6),
            (38, 7),
            (38, 8),
            (38, 9),
            (38, 10),
            (39, 0),
            (39, 1),
            (39, 2),
            (39, 3),
            (39, 4),
            (39, 5),
            (39, 6),
            (39, 7),
            (39, 8),
            (39, 9),
            (39, 10),
            (40, 0),
            (40, 1),
            (40, 2),
            (40, 3),
            (40, 4),
            (40, 5),
            (40, 6),
            (40, 7),
            (40, 8),
            (40, 9),
            (40, 10),
            (40, 11),
            (41, 0),
            (41, 1),
            (41, 2),
            (41, 3),
            (41, 4),
            (41, 5),
            (41, 6),
            (41, 7),
            (41, 8),
            (41, 9),
            (41, 10),
            (41, 11),
            (42, 0),
            (42, 1),
            (42, 2),
            (42, 3),
            (42, 4),
            (42, 5),
            (42, 6),
            (42, 7),
            (42, 8),
            (42, 9),
            (42, 10),
            (42, 11),
            (43, 0),
            (43, 1),
            (43, 2),
            (43, 3),
            (43, 4),
            (43, 5),
            (43, 6),
            (43, 7),
            (43, 8),
            (43, 9),
            (43, 10),
            (43, 11),
            (43, 12),
            (44, 0),
            (44, 1),
            (44, 2),
            (44, 3),
            (44, 4),
            (44, 5),
            (44, 6),
            (44, 7),
            (44, 8),
            (44, 9),
            (44, 10),
            (44, 11),
            (44, 12),
            (45, 0),
            (45, 1),
            (45, 2),
            (45, 3),
            (45, 4),
            (45, 5),
            (45, 6),
            (45, 7),
            (45, 8),
            (45, 9),
            (45, 10),
            (45, 11),
            (45, 12),
            (45, 13),
            (46, 0),
            (46, 1),
            (46, 2),
            (46, 3),
            (46, 4),
            (46, 5),
            (46, 6),
            (46, 7),
            (46, 8),
            (46, 9),
            (46, 10),
            (46, 11),
            (46, 12),
            (46, 13),
            (47, 0),
            (47, 1),
            (47, 2),
            (47, 3),
            (47, 4),
            (47, 5),
            (47, 6),
            (47, 7),
            (47, 8),
            (47, 9),
            (47, 10),
            (47, 11),
            (47, 12),
            (47, 13),
            (47, 14),
            (48, 0),
            (48, 1),
            (48, 2),
            (48, 3),
            (48, 4),
            (48, 5),
            (48, 6),
            (48, 7),
            (48, 8),
            (48, 9),
            (48, 10),
            (48, 11),
            (48, 12),
            (48, 13),
            (48, 14),
            (49, 0),
            (49, 1),
            (49, 2),
            (49, 3),
            (49, 4),
            (49, 5),
            (49, 6),
            (49, 7),
            (49, 8),
            (49, 9),
            (49, 10),
            (49, 11),
            (49, 12),
            (49, 13),
            (49, 14),
            (49, 15),
            (50, 0),
            (50, 1),
            (50, 2),
            (50, 3),
            (50, 4),
            (50, 5),
            (50, 6),
            (50, 7),
            (50, 8),
            (50, 9),
            (50, 10),
            (50, 11),
            (50, 12),
            (50, 13),
            (50, 14),
            (50, 15),
            (51, 0),
            (51, 1),
            (51, 2),
            (51, 3),
            (51, 4),
            (51, 5),
            (51, 6),
            (51, 7),
            (51, 8),
            (51, 9),
            (51, 10),
            (51, 11),
            (51, 12),
            (51, 13),
            (51, 14),
            (51, 15),
            (52, 0),
            (52, 1),
            (52, 2),
            (52, 3),
            (52, 4),
            (52, 5),
            (52, 6),
            (52, 7),
            (52, 8),
            (52, 9),
            (52, 10),
            (52, 11),
            (52, 12),
            (52, 13),
            (52, 14),
            (52, 15),
            (52, 16),
            (53, 0),
            (53, 1),
            (53, 2),
            (53, 3),
            (53, 4),
            (53, 5),
            (53, 6),
            (53, 7),
            (53, 8),
            (53, 9),
            (53, 10),
            (53, 11),
            (53, 12),
            (53, 13),
            (53, 14),
            (53, 15),
            (53, 16),
            (54, 0),
            (54, 1),
            (54, 2),
            (54, 3),
            (54, 4),
            (54, 5),
            (54, 6),
            (54, 7),
            (54, 8),
            (54, 9),
            (54, 10),
            (54, 11),
            (54, 12),
            (54, 13),
            (54, 14),
            (54, 15),
            (54, 16),
            (54, 17),
            (55, 0),
            (55, 1),
            (55, 2),
            (55, 3),
            (55, 4),
            (55, 5),
            (55, 6),
            (55, 7),
            (55, 8),
            (55, 9),
            (55, 10),
            (55, 11),
            (55, 12),
            (55, 13),
            (55, 14),
            (55, 15),
            (55, 16),
            (55, 17),
            (56, 0),
            (56, 1),
            (56, 2),
            (56, 3),
            (56, 4),
            (56, 5),
            (56, 6),
            (56, 7),
            (56, 8),
            (56, 9),
            (56, 10),
            (56, 11),
            (56, 12),
            (56, 13),
            (56, 14),
            (56, 15),
            (56, 16),
            (56, 17),
            (56, 18),
            (57, 0),
            (57, 1),
            (57, 2),
            (57, 3),
            (57, 4),
            (57, 5),
            (57, 6),
            (57, 7),
            (57, 8),
            (57, 9),
            (57, 10),
            (57, 11),
            (57, 12),
            (57, 13),
            (57, 14),
            (57, 15),
            (57, 16),
            (57, 17),
            (57, 18),
            (58, 0),
            (58, 1),
            (58, 2),
            (58, 3),
            (58, 4),
            (58, 5),
            (58, 6),
            (58, 7),
            (58, 8),
            (58, 9),
            (58, 10),
            (58, 11),
            (58, 12),
            (58, 13),
            (58, 14),
            (58, 15),
            (58, 16),
            (58, 17),
            (58, 18),
            (59, 0),
            (59, 1),
            (59, 2),
            (59, 3),
            (59, 4),
            (59, 5),
            (59, 6),
            (59, 7),
            (59, 8),
            (59, 9),
            (59, 10),
            (59, 11),
            (59, 12),
            (59, 13),
            (59, 14),
            (59, 15),
            (59, 16),
            (59, 17),
            (59, 18),
            (59, 19),
            (60, 0),
            (60, 1),
            (60, 2),
            (60, 3),
            (60, 4),
            (60, 5),
            (60, 6),
            (60, 7),
            (60, 8),
            (60, 9),
            (60, 10),
            (60, 11),
            (60, 12),
            (60, 13),
            (60, 14),
            (60, 15),
            (60, 16),
            (60, 17),
            (60, 18),
            (60, 19),
            (61, 0),
            (61, 1),
            (61, 2),
            (61, 3),
            (61, 4),
            (61, 5),
            (61, 6),
            (61, 7),
            (61, 8),
            (61, 9),
            (61, 10),
            (61, 11),
            (61, 12),
            (61, 13),
            (61, 14),
            (61, 15),
            (61, 16),
            (61, 17),
            (61, 18),
            (61, 19),
            (61, 20),
            (62, 0),
            (62, 1),
            (62, 2),
            (62, 3),
            (62, 4),
            (62, 5),
            (62, 6),
            (62, 7),
            (62, 8),
            (62, 9),
            (62, 10),
            (62, 11),
            (62, 12),
            (62, 13),
            (62, 14),
            (62, 15),
            (62, 16),
            (62, 17),
            (62, 18),
            (62, 19),
            (62, 20),
            (63, 0),
            (63, 1),
            (63, 2),
            (63, 3),
            (63, 4),
            (63, 5),
            (63, 6),
            (63, 7),
            (63, 8),
            (63, 9),
            (63, 10),
            (63, 11),
            (63, 12),
            (63, 13),
            (63, 14),
            (63, 15),
            (63, 16),
            (63, 17),
            (63, 18),
            (63, 19),
            (63, 20),
            (63, 21),
            (64, 0),
            (64, 1),
            (64, 2),
            (64, 3),
            (64, 4),
            (64, 5),
            (64, 6),
            (64, 7),
            (64, 8),
            (64, 9),
            (64, 10),
            (64, 11),
            (64, 12),
            (64, 13),
            (64, 14),
            (64, 15),
            (64, 16),
            (64, 17),
            (64, 18),
            (64, 19),
            (64, 20),
            (64, 21),
            (65, 0),
            (65, 1),
            (65, 2),
            (65, 3),
            (65, 4),
            (65, 5),
            (65, 6),
            (65, 7),
            (65, 8),
            (65, 9),
            (65, 10),
            (65, 11),
            (65, 12),
            (65, 13),
            (65, 14),
            (65, 15),
            (65, 16),
            (65, 17),
            (65, 18),
            (65, 19),
            (65, 20),
            (65, 21),
            (65, 22),
            (66, 0),
            (66, 1),
            (66, 2),
            (66, 3),
            (66, 4),
            (66, 5),
            (66, 6),
            (66, 7),
            (66, 8),
            (66, 9),
            (66, 10),
            (66, 11),
            (66, 12),
            (66, 13),
            (66, 14),
            (66, 15),
            (66, 16),
            (66, 17),
            (66, 18),
            (66, 19),
            (66, 20),
            (66, 21),
            (66, 22),
            (67, 0),
            (67, 1),
            (67, 2),
            (67, 3),
            (67, 4),
            (67, 5),
            (67, 6),
            (67, 7),
            (67, 8),
            (67, 9),
            (67, 10),
            (67, 11),
            (67, 12),
            (67, 13),
            (67, 14),
            (67, 15),
            (67, 16),
            (67, 17),
            (67, 18),
            (67, 19),
            (67, 20),
            (67, 21),
            (67, 22),
            (68, 0),
            (68, 1),
            (68, 2),
            (68, 3),
            (68, 4),
            (68, 5),
            (68, 6),
            (68, 7),
            (68, 8),
            (68, 9),
            (68, 10),
            (68, 11),
            (68, 12),
            (68, 13),
            (68, 14),
            (68, 15),
            (68, 16),
            (68, 17),
            (68, 18),
            (68, 19),
            (68, 20),
            (68, 21),
            (68, 22),
            (68, 23),
            (69, 0),
            (69, 1),
            (69, 2),
            (69, 3),
            (69, 4),
            (69, 5),
            (69, 6),
            (69, 7),
            (69, 8),
            (69, 9),
            (69, 10),
            (69, 11),
            (69, 12),
            (69, 13),
            (69, 14),
            (69, 15),
            (69, 16),
            (69, 17),
            (69, 18),
            (69, 19),
            (69, 20),
            (69, 21),
            (69, 22),
            (69, 23),
            (70, 0),
            (70, 1),
            (70, 2),
            (70, 3),
            (70, 4),
            (70, 5),
            (70, 6),
            (70, 7),
            (70, 8),
            (70, 9),
            (70, 10),
            (70, 11),
            (70, 12),
            (70, 13),
            (70, 14),
            (70, 15),
            (70, 16),
            (70, 17),
            (70, 18),
            (70, 19),
            (70, 20),
            (70, 21),
            (70, 22),
            (70, 23),
            (70, 24),
            (71, 0),
            (71, 1),
            (71, 2),
            (71, 3),
            (71, 4),
            (71, 5),
            (71, 6),
            (71, 7),
            (71, 8),
            (71, 9),
            (71, 10),
            (71, 11),
            (71, 12),
            (71, 13),
            (71, 14),
            (71, 15),
            (71, 16),
            (71, 17),
            (71, 18),
            (71, 19),
            (71, 20),
            (71, 21),
            (71, 22),
            (71, 23),
            (71, 24),
            (72, 0),
            (72, 1),
            (72, 2),
            (72, 3),
            (72, 4),
            (72, 5),
            (72, 6),
            (72, 7),
            (72, 8),
            (72, 9),
            (72, 10),
            (72, 11),
            (72, 12),
            (72, 13),
            (72, 14),
            (72, 15),
            (72, 16),
            (72, 17),
            (72, 18),
            (72, 19),
            (72, 20),
            (72, 21),
            (72, 22),
            (72, 23),
            (72, 24),
            (72, 25),
            (73, 0),
            (73, 1),
            (73, 2),
            (73, 3),
            (73, 4),
            (73, 5),
            (73, 6),
            (73, 7),
            (73, 8),
            (73, 9),
            (73, 10),
            (73, 11),
            (73, 12),
            (73, 13),
            (73, 14),
            (73, 15),
            (73, 16),
            (73, 17),
            (73, 18),
            (73, 19),
            (73, 20),
            (73, 21),
            (73, 22),
            (73, 23),
            (73, 24),
            (73, 25),
            (74, 0),
            (74, 1),
            (74, 2),
            (74, 3),
            (74, 4),
            (74, 5),
            (74, 6),
            (74, 7),
            (74, 8),
            (74, 9),
            (74, 10),
            (74, 11),
            (74, 12),
            (74, 13),
            (74, 14),
            (74, 15),
            (74, 16),
            (74, 17),
            (74, 18),
            (74, 19),
            (74, 20),
            (74, 21),
            (74, 22),
            (74, 23),
            (74, 24),
            (74, 25),
            (75, 0),
            (75, 1),
            (75, 2),
            (75, 3),
            (75, 4),
            (75, 5),
            (75, 6),
            (75, 7),
            (75, 8),
            (75, 9),
            (75, 10),
            (75, 11),
            (75, 12),
            (75, 13),
            (75, 14),
            (75, 15),
            (75, 16),
            (75, 17),
            (75, 18),
            (75, 19),
            (75, 20),
            (75, 21),
            (75, 22),
            (75, 23),
            (75, 24),
            (75, 25),
            (75, 26),
            (76, 0),
            (76, 1),
            (76, 2),
            (76, 3),
            (76, 4),
            (76, 5),
            (76, 6),
            (76, 7),
            (76, 8),
            (76, 9),
            (76, 10),
            (76, 11),
            (76, 12),
            (76, 13),
            (76, 14),
            (76, 15),
            (76, 16),
            (76, 17),
            (76, 18),
            (76, 19),
            (76, 20),
            (76, 21),
            (76, 22),
            (76, 23),
            (76, 24),
            (76, 25),
            (76, 26),
            (77, 0),
            (77, 1),
            (77, 2),
            (77, 3),
            (77, 4),
            (77, 5),
            (77, 6),
            (77, 7),
            (77, 8),
            (77, 9),
            (77, 10),
            (77, 11),
            (77, 12),
            (77, 13),
            (77, 14),
            (77, 15),
            (77, 16),
            (77, 17),
            (77, 18),
            (77, 19),
            (77, 20),
            (77, 21),
            (77, 22),
            (77, 23),
            (77, 24),
            (77, 25),
            (77, 26),
            (77, 27),
            (78, 0),
            (78, 1),
            (78, 2),
            (78, 3),
            (78, 4),
            (78, 5),
            (78, 6),
            (78, 7),
            (78, 8),
            (78, 9),
            (78, 10),
            (78, 11),
            (78, 12),
            (78, 13),
            (78, 14),
            (78, 15),
            (78, 16),
            (78, 17),
            (78, 18),
            (78, 19),
            (78, 20),
            (78, 21),
            (78, 22),
            (78, 23),
            (78, 24),
            (78, 25),
            (78, 26),
            (78, 27),
            (79, 0),
            (79, 1),
            (79, 2),
            (79, 3),
            (79, 4),
            (79, 5),
            (79, 6),
            (79, 7),
            (79, 8),
            (79, 9),
            (79, 10),
            (79, 11),
            (79, 12),
            (79, 13),
            (79, 14),
            (79, 15),
            (79, 16),
            (79, 17),
            (79, 18),
            (79, 19),
            (79, 20),
            (79, 21),
            (79, 22),
            (79, 23),
            (79, 24),
            (79, 25),
            (79, 26),
            (79, 27),
            (79, 28),
            (80, 0),
            (80, 1),
            (80, 2),
            (80, 3),
            (80, 4),
            (80, 5),
            (80, 6),
            (80, 7),
            (80, 8),
            (80, 9),
            (80, 10),
            (80, 11),
            (80, 12),
            (80, 13),
            (80, 14),
            (80, 15),
            (80, 16),
            (80, 17),
            (80, 18),
            (80, 19),
            (80, 20),
            (80, 21),
            (80, 22),
            (80, 23),
            (80, 24),
            (80, 25),
            (80, 26),
            (80, 27),
            (80, 28),
            (81, 0),
            (81, 1),
            (81, 2),
            (81, 3),
            (81, 4),
            (81, 5),
            (81, 6),
            (81, 7),
            (81, 8),
            (81, 9),
            (81, 10),
            (81, 11),
            (81, 12),
            (81, 13),
            (81, 14),
            (81, 15),
            (81, 16),
            (81, 17),
            (81, 18),
            (81, 19),
            (81, 20),
            (81, 21),
            (81, 22),
            (81, 23),
            (81, 24),
            (81, 25),
            (81, 26),
            (81, 27),
            (81, 28),
            (81, 29),
            (82, 0),
            (82, 1),
            (82, 2),
            (82, 3),
            (82, 4),
            (82, 5),
            (82, 6),
            (82, 7),
            (82, 8),
            (82, 9),
            (82, 10),
            (82, 11),
            (82, 12),
            (82, 13),
            (82, 14),
            (82, 15),
            (82, 16),
            (82, 17),
            (82, 18),
            (82, 19),
            (82, 20),
            (82, 21),
            (82, 22),
            (82, 23),
            (82, 24),
            (82, 25),
            (82, 26),
            (82, 27),
            (82, 28),
            (82, 29),
            (83, 0),
            (83, 1),
            (83, 2),
            (83, 3),
            (83, 4),
            (83, 5),
            (83, 6),
            (83, 7),
            (83, 8),
            (83, 9),
            (83, 10),
            (83, 11),
            (83, 12),
            (83, 13),
            (83, 14),
            (83, 15),
            (83, 16),
            (83, 17),
            (83, 18),
            (83, 19),
            (83, 20),
            (83, 21),
            (83, 22),
            (83, 23),
            (83, 24),
            (83, 25),
            (83, 26),
            (83, 27),
            (83, 28),
            (83, 29),
            (84, 0),
            (84, 1),
            (84, 2),
            (84, 3),
            (84, 4),
            (84, 5),
            (84, 6),
            (84, 7),
            (84, 8),
            (84, 9),
            (84, 10),
            (84, 11),
            (84, 12),
            (84, 13),
            (84, 14),
            (84, 15),
            (84, 16),
            (84, 17),
            (84, 18),
            (84, 19),
            (84, 20),
            (84, 21),
            (84, 22),
            (84, 23),
            (84, 24),
            (84, 25),
            (84, 26),
            (84, 27),
            (84, 28),
            (84, 29),
            (84, 30),
            (85, 0),
            (85, 1),
            (85, 2),
            (85, 3),
            (85, 4),
            (85, 5),
            (85, 6),
            (85, 7),
            (85, 8),
            (85, 9),
            (85, 10),
            (85, 11),
            (85, 12),
            (85, 13),
            (85, 14),
            (85, 15),
            (85, 16),
            (85, 17),
            (85, 18),
            (85, 19),
            (85, 20),
            (85, 21),
            (85, 22),
            (85, 23),
            (85, 24),
            (85, 25),
            (85, 26),
            (85, 27),
            (85, 28),
            (85, 29),
            (85, 30),
            (86, 0),
            (86, 1),
            (86, 2),
            (86, 3),
            (86, 4),
            (86, 5),
            (86, 6),
            (86, 7),
            (86, 8),
            (86, 9),
            (86, 10),
            (86, 11),
            (86, 12),
            (86, 13),
            (86, 14),
            (86, 15),
            (86, 16),
            (86, 17),
            (86, 18),
            (86, 19),
            (86, 20),
            (86, 21),
            (86, 22),
            (86, 23),
            (86, 24),
            (86, 25),
            (86, 26),
            (86, 27),
            (86, 28),
            (86, 29),
            (86, 30),
            (86, 31),
            (87, 0),
            (87, 1),
            (87, 2),
            (87, 3),
            (87, 4),
            (87, 5),
            (87, 6),
            (87, 7),
            (87, 8),
            (87, 9),
            (87, 10),
            (87, 11),
            (87, 12),
            (87, 13),
            (87, 14),
            (87, 15),
            (87, 16),
            (87, 17),
            (87, 18),
            (87, 19),
            (87, 20),
            (87, 21),
            (87, 22),
            (87, 23),
            (87, 24),
            (87, 25),
            (87, 26),
            (87, 27),
            (87, 28),
            (87, 29),
            (87, 30),
            (87, 31),
            (88, 0),
            (88, 1),
            (88, 2),
            (88, 3),
            (88, 4),
            (88, 5),
            (88, 6),
            (88, 7),
            (88, 8),
            (88, 9),
            (88, 10),
            (88, 11),
            (88, 12),
            (88, 13),
            (88, 14),
            (88, 15),
            (88, 16),
            (88, 17),
            (88, 18),
            (88, 19),
            (88, 20),
            (88, 21),
            (88, 22),
            (88, 23),
            (88, 24),
            (88, 25),
            (88, 26),
            (88, 27),
            (88, 28),
            (88, 29),
            (88, 30),
            (88, 31),
            (88, 32),
            (89, 0),
            (89, 1),
            (89, 2),
            (89, 3),
            (89, 4),
            (89, 5),
            (89, 6),
            (89, 7),
            (89, 8),
            (89, 9),
            (89, 10),
            (89, 11),
            (89, 12),
            (89, 13),
            (89, 14),
            (89, 15),
            (89, 16),
            (89, 17),
            (89, 18),
            (89, 19),
            (89, 20),
            (89, 21),
            (89, 22),
            (89, 23),
            (89, 24),
            (89, 25),
            (89, 26),
            (89, 27),
            (89, 28),
            (89, 29),
            (89, 30),
            (89, 31),
            (89, 32),
        }:
            return 1
        return 18

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_2_0_outputs, num_attn_0_1_outputs)
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


print(
    run(
        ["<s>", "0", "3", "3", "3", "1", "3", "2", "4", "0", "0", "4", "2", "1", "</s>"]
    )
)
