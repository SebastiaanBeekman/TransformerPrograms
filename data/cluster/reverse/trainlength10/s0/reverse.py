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
        "output/length/rasp/reverse/trainlength10/s0/reverse_weights.csv",
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
        if q_position in {0}:
            return k_position == 3
        elif q_position in {1, 2}:
            return k_position == 9
        elif q_position in {3}:
            return k_position == 6
        elif q_position in {4}:
            return k_position == 5
        elif q_position in {5, 6, 7}:
            return k_position == 2
        elif q_position in {8, 9}:
            return k_position == 1
        elif q_position in {10}:
            return k_position == 17
        elif q_position in {16, 11}:
            return k_position == 19
        elif q_position in {12}:
            return k_position == 10
        elif q_position in {19, 13}:
            return k_position == 15
        elif q_position in {14, 15}:
            return k_position == 12
        elif q_position in {17}:
            return k_position == 14
        elif q_position in {18}:
            return k_position == 16

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 2}:
            return k_position == 7
        elif q_position in {1, 3}:
            return k_position == 5
        elif q_position in {4}:
            return k_position == 0
        elif q_position in {5}:
            return k_position == 3
        elif q_position in {8, 6, 7}:
            return k_position == 1
        elif q_position in {9}:
            return k_position == 2
        elif q_position in {10}:
            return k_position == 19
        elif q_position in {11, 12}:
            return k_position == 16
        elif q_position in {16, 13}:
            return k_position == 13
        elif q_position in {14}:
            return k_position == 14
        elif q_position in {15}:
            return k_position == 10
        elif q_position in {17}:
            return k_position == 17
        elif q_position in {18}:
            return k_position == 18
        elif q_position in {19}:
            return k_position == 15

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0}:
            return k_position == 4
        elif q_position in {1}:
            return k_position == 8
        elif q_position in {2}:
            return k_position == 7
        elif q_position in {3}:
            return k_position == 0
        elif q_position in {4}:
            return k_position == 2
        elif q_position in {8, 9, 5, 6}:
            return k_position == 1
        elif q_position in {7}:
            return k_position == 9
        elif q_position in {10}:
            return k_position == 14
        elif q_position in {11}:
            return k_position == 12
        elif q_position in {18, 12}:
            return k_position == 15
        elif q_position in {13, 15}:
            return k_position == 19
        elif q_position in {14}:
            return k_position == 13
        elif q_position in {16}:
            return k_position == 10
        elif q_position in {17, 19}:
            return k_position == 11

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 6}:
            return k_position == 3
        elif q_position in {1}:
            return k_position == 8
        elif q_position in {2}:
            return k_position == 7
        elif q_position in {3}:
            return k_position == 6
        elif q_position in {8, 4}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 4
        elif q_position in {7}:
            return k_position == 2
        elif q_position in {9}:
            return k_position == 1
        elif q_position in {16, 10}:
            return k_position == 11
        elif q_position in {11}:
            return k_position == 17
        elif q_position in {12}:
            return k_position == 15
        elif q_position in {13}:
            return k_position == 13
        elif q_position in {14}:
            return k_position == 12
        elif q_position in {17, 15}:
            return k_position == 16
        elif q_position in {18}:
            return k_position == 14
        elif q_position in {19}:
            return k_position == 19

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(position, token):
        if position in {0, 1, 2, 5, 7}:
            return token == "2"
        elif position in {3, 4, 9, 11, 12, 13, 16, 18, 19}:
            return token == ""
        elif position in {17, 10, 6, 14}:
            return token == "1"
        elif position in {8, 15}:
            return token == "4"

    attn_0_4_pattern = select_closest(tokens, positions, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(q_position, k_position):
        if q_position in {0, 4}:
            return k_position == 4
        elif q_position in {1}:
            return k_position == 7
        elif q_position in {2}:
            return k_position == 6
        elif q_position in {3}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 3
        elif q_position in {9, 6}:
            return k_position == 2
        elif q_position in {7}:
            return k_position == 9
        elif q_position in {8}:
            return k_position == 1
        elif q_position in {10, 11}:
            return k_position == 13
        elif q_position in {12, 13}:
            return k_position == 10
        elif q_position in {14}:
            return k_position == 19
        elif q_position in {15}:
            return k_position == 17
        elif q_position in {16}:
            return k_position == 14
        elif q_position in {17, 18}:
            return k_position == 11
        elif q_position in {19}:
            return k_position == 15

    attn_0_5_pattern = select_closest(positions, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(q_position, k_position):
        if q_position in {0, 3}:
            return k_position == 4
        elif q_position in {1}:
            return k_position == 8
        elif q_position in {2}:
            return k_position == 5
        elif q_position in {4, 5, 7}:
            return k_position == 2
        elif q_position in {6}:
            return k_position == 3
        elif q_position in {8, 9}:
            return k_position == 1
        elif q_position in {16, 17, 10, 18}:
            return k_position == 13
        elif q_position in {11, 15}:
            return k_position == 15
        elif q_position in {12, 14}:
            return k_position == 11
        elif q_position in {13}:
            return k_position == 17
        elif q_position in {19}:
            return k_position == 14

    attn_0_6_pattern = select_closest(positions, positions, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(q_position, k_position):
        if q_position in {0}:
            return k_position == 3
        elif q_position in {1}:
            return k_position == 6
        elif q_position in {2}:
            return k_position == 7
        elif q_position in {3, 4}:
            return k_position == 0
        elif q_position in {5, 6, 7, 8, 9}:
            return k_position == 1
        elif q_position in {10}:
            return k_position == 14
        elif q_position in {11}:
            return k_position == 12
        elif q_position in {12}:
            return k_position == 19
        elif q_position in {13, 15}:
            return k_position == 18
        elif q_position in {16, 14}:
            return k_position == 10
        elif q_position in {17, 19}:
            return k_position == 13
        elif q_position in {18}:
            return k_position == 11

    attn_0_7_pattern = select_closest(positions, positions, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(token, position):
        if token in {"</s>", "0", "1", "4", "2", "<s>"}:
            return position == 9
        elif token in {"3"}:
            return position == 8

    num_attn_0_0_pattern = select(positions, tokens, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(token, position):
        if token in {"0"}:
            return position == 18
        elif token in {"</s>", "1", "3", "<s>"}:
            return position == 8
        elif token in {"2"}:
            return position == 15
        elif token in {"4"}:
            return position == 16

    num_attn_0_1_pattern = select(positions, tokens, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(token, position):
        if token in {"3", "0", "</s>", "1", "4", "2", "<s>"}:
            return position == 9

    num_attn_0_2_pattern = select(positions, tokens, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(token, position):
        if token in {"3", "0", "</s>", "1", "4", "<s>"}:
            return position == 9
        elif token in {"2"}:
            return position == 8

    num_attn_0_3_pattern = select(positions, tokens, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(token, position):
        if token in {"3", "</s>", "0", "2", "<s>"}:
            return position == 9
        elif token in {"1"}:
            return position == 7
        elif token in {"4"}:
            return position == 8

    num_attn_0_4_pattern = select(positions, tokens, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(q_position, k_position):
        if q_position in {0}:
            return k_position == 6
        elif q_position in {
            1,
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
        }:
            return k_position == 9

    num_attn_0_5_pattern = select(positions, positions, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(token, position):
        if token in {"0"}:
            return position == 8
        elif token in {"</s>", "1", "4", "<s>"}:
            return position == 9
        elif token in {"2"}:
            return position == 13
        elif token in {"3"}:
            return position == 7

    num_attn_0_6_pattern = select(positions, tokens, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(token, position):
        if token in {"0", "2", "3"}:
            return position == 8
        elif token in {"1", "4"}:
            return position == 7
        elif token in {"</s>"}:
            return position == 9
        elif token in {"<s>"}:
            return position == 6

    num_attn_0_7_pattern = select(positions, tokens, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_6_output, attn_0_7_output):
        key = (attn_0_6_output, attn_0_7_output)
        if key in {
            ("1", "1"),
            ("1", "2"),
            ("1", "3"),
            ("1", "4"),
            ("1", "</s>"),
            ("2", "1"),
            ("2", "3"),
            ("2", "</s>"),
            ("3", "1"),
            ("3", "3"),
            ("3", "4"),
            ("3", "</s>"),
            ("</s>", "1"),
            ("</s>", "3"),
            ("</s>", "4"),
            ("</s>", "</s>"),
        }:
            return 16
        elif key in {
            ("0", "0"),
            ("0", "1"),
            ("0", "2"),
            ("0", "3"),
            ("0", "4"),
            ("0", "</s>"),
            ("0", "<s>"),
            ("1", "0"),
            ("2", "0"),
            ("3", "0"),
            ("4", "0"),
            ("</s>", "0"),
            ("<s>", "0"),
        }:
            return 19
        return 17

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_6_outputs, attn_0_7_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_6_output, attn_0_7_output):
        key = (attn_0_6_output, attn_0_7_output)
        if key in {
            ("0", "0"),
            ("0", "2"),
            ("0", "3"),
            ("1", "<s>"),
            ("2", "0"),
            ("2", "2"),
            ("2", "3"),
            ("3", "0"),
            ("3", "2"),
            ("3", "3"),
            ("3", "</s>"),
            ("<s>", "0"),
            ("<s>", "2"),
            ("<s>", "3"),
            ("<s>", "</s>"),
        }:
            return 14
        elif key in {
            ("1", "0"),
            ("1", "1"),
            ("1", "2"),
            ("1", "3"),
            ("1", "</s>"),
            ("</s>", "0"),
            ("</s>", "1"),
            ("</s>", "2"),
            ("</s>", "3"),
            ("</s>", "</s>"),
        }:
            return 10
        elif key in {("0", "4"), ("1", "4"), ("2", "4"), ("4", "4"), ("</s>", "4")}:
            return 8
        elif key in {("0", "</s>"), ("2", "</s>")}:
            return 0
        elif key in {("4", "0")}:
            return 16
        return 2

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_6_outputs, attn_0_7_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_6_output, num_attn_0_7_output):
        key = (num_attn_0_6_output, num_attn_0_7_output)
        return 15

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_6_outputs, num_attn_0_7_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_4_output, num_attn_0_1_output):
        key = (num_attn_0_4_output, num_attn_0_1_output)
        return 17

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_4_outputs, num_attn_0_1_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_position, k_position):
        if q_position in {0, 3}:
            return k_position == 4
        elif q_position in {1}:
            return k_position == 6
        elif q_position in {2}:
            return k_position == 0
        elif q_position in {8, 4, 6, 7}:
            return k_position == 1
        elif q_position in {5}:
            return k_position == 3
        elif q_position in {9}:
            return k_position == 7
        elif q_position in {17, 10, 15}:
            return k_position == 16
        elif q_position in {11}:
            return k_position == 19
        elif q_position in {12}:
            return k_position == 13
        elif q_position in {13}:
            return k_position == 12
        elif q_position in {19, 14}:
            return k_position == 18
        elif q_position in {16}:
            return k_position == 17
        elif q_position in {18}:
            return k_position == 5

    attn_1_0_pattern = select_closest(positions, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, tokens)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_position, k_position):
        if q_position in {0, 11, 12, 13, 14, 15, 17, 18, 19}:
            return k_position == 9
        elif q_position in {1, 3, 7}:
            return k_position == 3
        elif q_position in {2}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 6
        elif q_position in {5}:
            return k_position == 7
        elif q_position in {9, 6}:
            return k_position == 8
        elif q_position in {8}:
            return k_position == 1
        elif q_position in {10}:
            return k_position == 19
        elif q_position in {16}:
            return k_position == 17

    attn_1_1_pattern = select_closest(positions, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_7_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(q_position, k_position):
        if q_position in {0, 6}:
            return k_position == 1
        elif q_position in {1}:
            return k_position == 7
        elif q_position in {9, 2, 4}:
            return k_position == 4
        elif q_position in {3}:
            return k_position == 14
        elif q_position in {5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}:
            return k_position == 8
        elif q_position in {7}:
            return k_position == 2
        elif q_position in {8}:
            return k_position == 3

    attn_1_2_pattern = select_closest(positions, positions, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, tokens)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(q_position, k_position):
        if q_position in {0, 1, 18}:
            return k_position == 4
        elif q_position in {2, 3, 4}:
            return k_position == 2
        elif q_position in {8, 5}:
            return k_position == 1
        elif q_position in {6}:
            return k_position == 8
        elif q_position in {7}:
            return k_position == 9
        elif q_position in {9}:
            return k_position == 7
        elif q_position in {16, 10, 13}:
            return k_position == 12
        elif q_position in {11}:
            return k_position == 13
        elif q_position in {12}:
            return k_position == 16
        elif q_position in {14}:
            return k_position == 11
        elif q_position in {15}:
            return k_position == 19
        elif q_position in {17}:
            return k_position == 3
        elif q_position in {19}:
            return k_position == 18

    attn_1_3_pattern = select_closest(positions, positions, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, tokens)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(q_position, k_position):
        if q_position in {0, 3}:
            return k_position == 7
        elif q_position in {1, 5}:
            return k_position == 5
        elif q_position in {8, 2, 4}:
            return k_position == 2
        elif q_position in {9, 6}:
            return k_position == 8
        elif q_position in {7}:
            return k_position == 3
        elif q_position in {10, 12}:
            return k_position == 18
        elif q_position in {11, 13, 16, 17, 18}:
            return k_position == 12
        elif q_position in {14}:
            return k_position == 17
        elif q_position in {15}:
            return k_position == 14
        elif q_position in {19}:
            return k_position == 19

    attn_1_4_pattern = select_closest(positions, positions, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, attn_0_5_outputs)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(q_position, k_position):
        if q_position in {0, 1}:
            return k_position == 4
        elif q_position in {2, 5, 6}:
            return k_position == 8
        elif q_position in {3, 4}:
            return k_position == 7
        elif q_position in {7}:
            return k_position == 2
        elif q_position in {8}:
            return k_position == 3
        elif q_position in {9}:
            return k_position == 9
        elif q_position in {19, 10, 18}:
            return k_position == 17
        elif q_position in {11}:
            return k_position == 15
        elif q_position in {12}:
            return k_position == 10
        elif q_position in {16, 13}:
            return k_position == 11
        elif q_position in {17, 14}:
            return k_position == 16
        elif q_position in {15}:
            return k_position == 14

    attn_1_5_pattern = select_closest(positions, positions, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, attn_0_3_outputs)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(q_position, k_position):
        if q_position in {0, 8}:
            return k_position == 3
        elif q_position in {1, 7}:
            return k_position == 4
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 6
        elif q_position in {9, 4, 6}:
            return k_position == 8
        elif q_position in {5}:
            return k_position == 5
        elif q_position in {10}:
            return k_position == 13
        elif q_position in {11, 14}:
            return k_position == 9
        elif q_position in {12}:
            return k_position == 11
        elif q_position in {13}:
            return k_position == 12
        elif q_position in {15}:
            return k_position == 17
        elif q_position in {16}:
            return k_position == 16
        elif q_position in {17}:
            return k_position == 10
        elif q_position in {18, 19}:
            return k_position == 19

    attn_1_6_pattern = select_closest(positions, positions, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, attn_0_5_outputs)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(q_position, k_position):
        if q_position in {0, 2, 3, 4, 5}:
            return k_position == 2
        elif q_position in {1, 15}:
            return k_position == 5
        elif q_position in {6}:
            return k_position == 8
        elif q_position in {16, 7}:
            return k_position == 9
        elif q_position in {8}:
            return k_position == 0
        elif q_position in {9}:
            return k_position == 1
        elif q_position in {10, 12}:
            return k_position == 3
        elif q_position in {11}:
            return k_position == 19
        elif q_position in {13}:
            return k_position == 14
        elif q_position in {14}:
            return k_position == 15
        elif q_position in {17, 18}:
            return k_position == 11
        elif q_position in {19}:
            return k_position == 18

    attn_1_7_pattern = select_closest(positions, positions, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, attn_0_1_outputs)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(attn_0_5_output, attn_0_6_output):
        if attn_0_5_output in {"0", "2"}:
            return attn_0_6_output == "1"
        elif attn_0_5_output in {"1"}:
            return attn_0_6_output == "<pad>"
        elif attn_0_5_output in {"</s>", "3", "<s>"}:
            return attn_0_6_output == ""
        elif attn_0_5_output in {"4"}:
            return attn_0_6_output == "4"

    num_attn_1_0_pattern = select(attn_0_6_outputs, attn_0_5_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_1_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(q_attn_0_3_output, k_attn_0_3_output):
        if q_attn_0_3_output in {"0", "2"}:
            return k_attn_0_3_output == "1"
        elif q_attn_0_3_output in {"4", "1", "3", "<s>"}:
            return k_attn_0_3_output == ""
        elif q_attn_0_3_output in {"</s>"}:
            return k_attn_0_3_output == "0"

    num_attn_1_1_pattern = select(attn_0_3_outputs, attn_0_3_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_5_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(q_attn_0_3_output, k_attn_0_3_output):
        if q_attn_0_3_output in {"0", "2"}:
            return k_attn_0_3_output == "<pad>"
        elif q_attn_0_3_output in {"1"}:
            return k_attn_0_3_output == "4"
        elif q_attn_0_3_output in {"3"}:
            return k_attn_0_3_output == "2"
        elif q_attn_0_3_output in {"4", "<s>"}:
            return k_attn_0_3_output == ""
        elif q_attn_0_3_output in {"</s>"}:
            return k_attn_0_3_output == "1"

    num_attn_1_2_pattern = select(attn_0_3_outputs, attn_0_3_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_5_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(q_attn_0_3_output, k_attn_0_3_output):
        if q_attn_0_3_output in {"0"}:
            return k_attn_0_3_output == "4"
        elif q_attn_0_3_output in {"1"}:
            return k_attn_0_3_output == "2"
        elif q_attn_0_3_output in {"2"}:
            return k_attn_0_3_output == "<pad>"
        elif q_attn_0_3_output in {"4", "3", "<s>"}:
            return k_attn_0_3_output == ""
        elif q_attn_0_3_output in {"</s>"}:
            return k_attn_0_3_output == "0"

    num_attn_1_3_pattern = select(attn_0_3_outputs, attn_0_3_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_2_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(q_attn_0_3_output, k_attn_0_3_output):
        if q_attn_0_3_output in {"0", "2", "3", "<s>"}:
            return k_attn_0_3_output == ""
        elif q_attn_0_3_output in {"1"}:
            return k_attn_0_3_output == "0"
        elif q_attn_0_3_output in {"4"}:
            return k_attn_0_3_output == "3"
        elif q_attn_0_3_output in {"</s>"}:
            return k_attn_0_3_output == "1"

    num_attn_1_4_pattern = select(attn_0_3_outputs, attn_0_3_outputs, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_5_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(q_attn_0_3_output, k_attn_0_3_output):
        if q_attn_0_3_output in {"0", "<s>"}:
            return k_attn_0_3_output == ""
        elif q_attn_0_3_output in {"1"}:
            return k_attn_0_3_output == "0"
        elif q_attn_0_3_output in {"2", "4"}:
            return k_attn_0_3_output == "<pad>"
        elif q_attn_0_3_output in {"3"}:
            return k_attn_0_3_output == "4"
        elif q_attn_0_3_output in {"</s>"}:
            return k_attn_0_3_output == "3"

    num_attn_1_5_pattern = select(attn_0_3_outputs, attn_0_3_outputs, num_predicate_1_5)
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_5_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(q_attn_0_3_output, k_attn_0_3_output):
        if q_attn_0_3_output in {"0", "4"}:
            return k_attn_0_3_output == "<pad>"
        elif q_attn_0_3_output in {"1", "<s>"}:
            return k_attn_0_3_output == ""
        elif q_attn_0_3_output in {"2"}:
            return k_attn_0_3_output == "0"
        elif q_attn_0_3_output in {"3"}:
            return k_attn_0_3_output == "1"
        elif q_attn_0_3_output in {"</s>"}:
            return k_attn_0_3_output == "2"

    num_attn_1_6_pattern = select(attn_0_3_outputs, attn_0_3_outputs, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_5_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(attn_0_1_output, attn_0_3_output):
        if attn_0_1_output in {"0"}:
            return attn_0_3_output == "0"
        elif attn_0_1_output in {"1"}:
            return attn_0_3_output == "<pad>"
        elif attn_0_1_output in {"3", "</s>", "4", "2", "<s>"}:
            return attn_0_3_output == ""

    num_attn_1_7_pattern = select(attn_0_3_outputs, attn_0_1_outputs, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_1_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_7_output, attn_0_5_output):
        key = (attn_1_7_output, attn_0_5_output)
        if key in {
            ("0", "<s>"),
            ("1", "<s>"),
            ("2", "<s>"),
            ("3", "<s>"),
            ("4", "<s>"),
            ("<s>", "<s>"),
        }:
            return 4
        elif key in {
            ("0", "</s>"),
            ("1", "</s>"),
            ("2", "</s>"),
            ("3", "</s>"),
            ("4", "</s>"),
            ("</s>", "0"),
            ("</s>", "1"),
            ("</s>", "2"),
            ("</s>", "3"),
            ("</s>", "4"),
            ("</s>", "</s>"),
            ("</s>", "<s>"),
            ("<s>", "0"),
            ("<s>", "1"),
            ("<s>", "2"),
            ("<s>", "3"),
            ("<s>", "4"),
            ("<s>", "</s>"),
        }:
            return 16
        elif key in {("0", "4"), ("1", "4"), ("2", "4"), ("3", "4"), ("4", "4")}:
            return 5
        elif key in {("0", "3"), ("1", "3"), ("2", "3"), ("3", "3"), ("4", "3")}:
            return 19
        return 13

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_7_outputs, attn_0_5_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_0_6_output, attn_1_1_output):
        key = (attn_0_6_output, attn_1_1_output)
        if key in {
            ("1", "0"),
            ("1", "1"),
            ("1", "2"),
            ("1", "3"),
            ("1", "4"),
            ("1", "<s>"),
            ("4", "0"),
            ("4", "1"),
            ("4", "2"),
            ("4", "3"),
            ("4", "4"),
            ("4", "<s>"),
        }:
            return 18
        elif key in {
            ("3", "0"),
            ("3", "1"),
            ("3", "2"),
            ("3", "3"),
            ("3", "4"),
            ("3", "<s>"),
            ("<s>", "3"),
        }:
            return 16
        elif key in {
            ("2", "0"),
            ("2", "1"),
            ("2", "2"),
            ("2", "3"),
            ("2", "4"),
            ("2", "<s>"),
        }:
            return 10
        elif key in {("2", "</s>"), ("3", "</s>")}:
            return 7
        return 13

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_0_6_outputs, attn_1_1_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_0_2_output):
        key = num_attn_0_2_output
        if key in {0}:
            return 18
        return 4

    num_mlp_1_0_outputs = [num_mlp_1_0(k0) for k0 in num_attn_0_2_outputs]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_1_output, num_attn_1_7_output):
        key = (num_attn_1_1_output, num_attn_1_7_output)
        return 18

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_1_7_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(position, token):
        if position in {0, 2, 4, 6, 9, 10, 11, 13, 14, 17, 18, 19}:
            return token == "3"
        elif position in {8, 1}:
            return token == "</s>"
        elif position in {3, 7, 12, 15, 16}:
            return token == "4"
        elif position in {5}:
            return token == "2"

    attn_2_0_pattern = select_closest(tokens, positions, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_0_3_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(position, token):
        if position in {0, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}:
            return token == "2"
        elif position in {1}:
            return token == "<s>"
        elif position in {8, 2}:
            return token == "</s>"
        elif position in {6}:
            return token == "4"

    attn_2_1_pattern = select_closest(tokens, positions, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_4_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(position, token):
        if position in {0, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19}:
            return token == "3"
        elif position in {1, 4, 6}:
            return token == "0"
        elif position in {8, 2, 3}:
            return token == "</s>"
        elif position in {5}:
            return token == "<s>"
        elif position in {7}:
            return token == "2"
        elif position in {14}:
            return token == ""

    attn_2_2_pattern = select_closest(tokens, positions, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_6_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(position, token):
        if position in {0, 9, 10, 11, 13, 14, 16, 17, 18}:
            return token == "4"
        elif position in {1, 5}:
            return token == "<s>"
        elif position in {8, 2, 3}:
            return token == "</s>"
        elif position in {4, 12}:
            return token == "1"
        elif position in {6}:
            return token == "2"
        elif position in {19, 7}:
            return token == "0"
        elif position in {15}:
            return token == ""

    attn_2_3_pattern = select_closest(tokens, positions, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_1_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(q_position, k_position):
        if q_position in {0}:
            return k_position == 16
        elif q_position in {1}:
            return k_position == 7
        elif q_position in {8, 2}:
            return k_position == 8
        elif q_position in {3}:
            return k_position == 6
        elif q_position in {4}:
            return k_position == 5
        elif q_position in {5, 6}:
            return k_position == 4
        elif q_position in {7}:
            return k_position == 1
        elif q_position in {9}:
            return k_position == 19
        elif q_position in {19, 10, 11, 13}:
            return k_position == 14
        elif q_position in {12}:
            return k_position == 17
        elif q_position in {16, 18, 14, 15}:
            return k_position == 13
        elif q_position in {17}:
            return k_position == 15

    attn_2_4_pattern = select_closest(positions, positions, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_1_5_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(q_position, k_position):
        if q_position in {0, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19}:
            return k_position == 3
        elif q_position in {1, 6}:
            return k_position == 8
        elif q_position in {2}:
            return k_position == 7
        elif q_position in {3}:
            return k_position == 0
        elif q_position in {4}:
            return k_position == 4
        elif q_position in {5}:
            return k_position == 5
        elif q_position in {7}:
            return k_position == 9
        elif q_position in {8}:
            return k_position == 1
        elif q_position in {14}:
            return k_position == 14

    attn_2_5_pattern = select_closest(positions, positions, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_1_5_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(position, token):
        if position in {0, 3, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}:
            return token == "3"
        elif position in {8, 1, 4, 6}:
            return token == "<s>"
        elif position in {2}:
            return token == "1"
        elif position in {5}:
            return token == "</s>"

    attn_2_6_pattern = select_closest(tokens, positions, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, attn_1_2_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(q_position, k_position):
        if q_position in {0, 9, 18, 17}:
            return k_position == 12
        elif q_position in {1, 6}:
            return k_position == 8
        elif q_position in {2}:
            return k_position == 2
        elif q_position in {8, 3, 7}:
            return k_position == 3
        elif q_position in {4}:
            return k_position == 6
        elif q_position in {5}:
            return k_position == 7
        elif q_position in {10}:
            return k_position == 14
        elif q_position in {11}:
            return k_position == 18
        elif q_position in {16, 12}:
            return k_position == 16
        elif q_position in {13}:
            return k_position == 9
        elif q_position in {14}:
            return k_position == 19
        elif q_position in {15}:
            return k_position == 17
        elif q_position in {19}:
            return k_position == 10

    attn_2_7_pattern = select_closest(positions, positions, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_0_1_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(q_attn_0_3_output, k_attn_0_3_output):
        if q_attn_0_3_output in {"0"}:
            return k_attn_0_3_output == "0"
        elif q_attn_0_3_output in {"</s>", "1"}:
            return k_attn_0_3_output == "1"
        elif q_attn_0_3_output in {"2"}:
            return k_attn_0_3_output == "<pad>"
        elif q_attn_0_3_output in {"4", "3"}:
            return k_attn_0_3_output == ""
        elif q_attn_0_3_output in {"<s>"}:
            return k_attn_0_3_output == "<s>"

    num_attn_2_0_pattern = select(attn_0_3_outputs, attn_0_3_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_2_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(q_attn_0_3_output, k_attn_0_3_output):
        if q_attn_0_3_output in {"0"}:
            return k_attn_0_3_output == "2"
        elif q_attn_0_3_output in {"2", "1", "4", "<s>"}:
            return k_attn_0_3_output == ""
        elif q_attn_0_3_output in {"</s>", "3"}:
            return k_attn_0_3_output == "3"

    num_attn_2_1_pattern = select(attn_0_3_outputs, attn_0_3_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_2_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(q_attn_0_3_output, k_attn_0_3_output):
        if q_attn_0_3_output in {"0", "<s>"}:
            return k_attn_0_3_output == "</s>"
        elif q_attn_0_3_output in {"1"}:
            return k_attn_0_3_output == "3"
        elif q_attn_0_3_output in {"2"}:
            return k_attn_0_3_output == "2"
        elif q_attn_0_3_output in {"4", "</s>", "3"}:
            return k_attn_0_3_output == ""

    num_attn_2_2_pattern = select(attn_0_3_outputs, attn_0_3_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_2_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(q_attn_0_3_output, k_attn_0_3_output):
        if q_attn_0_3_output in {"</s>", "0", "2", "4"}:
            return k_attn_0_3_output == ""
        elif q_attn_0_3_output in {"1"}:
            return k_attn_0_3_output == "1"
        elif q_attn_0_3_output in {"3"}:
            return k_attn_0_3_output == "0"
        elif q_attn_0_3_output in {"<s>"}:
            return k_attn_0_3_output == "<pad>"

    num_attn_2_3_pattern = select(attn_0_3_outputs, attn_0_3_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_2_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(q_attn_0_3_output, k_attn_0_3_output):
        if q_attn_0_3_output in {"0", "2", "1", "3"}:
            return k_attn_0_3_output == "4"
        elif q_attn_0_3_output in {"</s>", "4"}:
            return k_attn_0_3_output == ""
        elif q_attn_0_3_output in {"<s>"}:
            return k_attn_0_3_output == "<s>"

    num_attn_2_4_pattern = select(attn_0_3_outputs, attn_0_3_outputs, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_0_2_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(q_attn_0_3_output, k_attn_0_3_output):
        if q_attn_0_3_output in {"0", "2"}:
            return k_attn_0_3_output == "3"
        elif q_attn_0_3_output in {"1"}:
            return k_attn_0_3_output == "0"
        elif q_attn_0_3_output in {"3", "<s>"}:
            return k_attn_0_3_output == ""
        elif q_attn_0_3_output in {"4"}:
            return k_attn_0_3_output == "<pad>"
        elif q_attn_0_3_output in {"</s>"}:
            return k_attn_0_3_output == "2"

    num_attn_2_5_pattern = select(attn_0_3_outputs, attn_0_3_outputs, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_0_5_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(q_attn_0_3_output, k_attn_0_3_output):
        if q_attn_0_3_output in {"0"}:
            return k_attn_0_3_output == "3"
        elif q_attn_0_3_output in {"1"}:
            return k_attn_0_3_output == "</s>"
        elif q_attn_0_3_output in {"2", "3", "<s>"}:
            return k_attn_0_3_output == ""
        elif q_attn_0_3_output in {"4"}:
            return k_attn_0_3_output == "2"
        elif q_attn_0_3_output in {"</s>"}:
            return k_attn_0_3_output == "4"

    num_attn_2_6_pattern = select(attn_0_3_outputs, attn_0_3_outputs, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_0_5_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(q_mlp_1_0_output, k_mlp_1_0_output):
        if q_mlp_1_0_output in {0}:
            return k_mlp_1_0_output == 12
        elif q_mlp_1_0_output in {1, 19, 5}:
            return k_mlp_1_0_output == 13
        elif q_mlp_1_0_output in {9, 2, 6, 7}:
            return k_mlp_1_0_output == 0
        elif q_mlp_1_0_output in {11, 3, 15}:
            return k_mlp_1_0_output == 6
        elif q_mlp_1_0_output in {17, 18, 4}:
            return k_mlp_1_0_output == 11
        elif q_mlp_1_0_output in {8}:
            return k_mlp_1_0_output == 18
        elif q_mlp_1_0_output in {10, 12}:
            return k_mlp_1_0_output == 8
        elif q_mlp_1_0_output in {13}:
            return k_mlp_1_0_output == 16
        elif q_mlp_1_0_output in {14}:
            return k_mlp_1_0_output == 1
        elif q_mlp_1_0_output in {16}:
            return k_mlp_1_0_output == 17

    num_attn_2_7_pattern = select(mlp_1_0_outputs, mlp_1_0_outputs, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_0_7_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_0_3_output, attn_0_6_output):
        key = (attn_0_3_output, attn_0_6_output)
        if key in {
            ("0", "4"),
            ("0", "</s>"),
            ("0", "<s>"),
            ("1", "<s>"),
            ("2", "4"),
            ("2", "</s>"),
            ("2", "<s>"),
            ("3", "<s>"),
            ("4", "4"),
            ("4", "</s>"),
            ("4", "<s>"),
        }:
            return 12
        elif key in {
            ("0", "0"),
            ("0", "1"),
            ("0", "2"),
            ("1", "0"),
            ("1", "1"),
            ("1", "2"),
            ("1", "</s>"),
            ("2", "0"),
            ("2", "1"),
            ("2", "2"),
            ("3", "0"),
            ("3", "1"),
            ("3", "2"),
            ("4", "0"),
            ("4", "1"),
            ("4", "2"),
        }:
            return 4
        elif key in {("0", "3"), ("1", "3"), ("2", "3"), ("3", "3"), ("4", "3")}:
            return 16
        return 11

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_0_3_outputs, attn_0_6_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_0_6_output, attn_0_7_output):
        key = (attn_0_6_output, attn_0_7_output)
        if key in {("2", "<s>"), ("4", "<s>"), ("</s>", "<s>")}:
            return 17
        elif key in {
            ("0", "</s>"),
            ("2", "</s>"),
            ("3", "</s>"),
            ("4", "</s>"),
            ("</s>", "0"),
            ("</s>", "2"),
            ("</s>", "3"),
            ("</s>", "4"),
            ("</s>", "</s>"),
            ("<s>", "</s>"),
        }:
            return 5
        elif key in {
            ("0", "0"),
            ("0", "1"),
            ("0", "2"),
            ("0", "3"),
            ("0", "4"),
            ("0", "<s>"),
            ("3", "<s>"),
            ("<s>", "3"),
            ("<s>", "<s>"),
        }:
            return 0
        elif key in {("</s>", "1")}:
            return 14
        return 18

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_0_6_outputs, attn_0_7_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_2_2_output, num_attn_1_5_output):
        key = (num_attn_2_2_output, num_attn_1_5_output)
        if key in {
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (0, 8),
            (0, 9),
            (0, 10),
            (0, 11),
            (0, 12),
            (0, 13),
            (0, 14),
            (0, 15),
            (0, 16),
            (0, 17),
            (0, 18),
            (0, 19),
            (0, 20),
            (0, 21),
            (0, 22),
            (0, 23),
            (0, 24),
            (0, 25),
            (0, 26),
            (0, 27),
            (0, 28),
            (0, 29),
            (0, 30),
            (0, 31),
            (0, 32),
            (0, 33),
            (0, 34),
            (0, 35),
            (0, 36),
            (0, 37),
            (0, 38),
            (0, 39),
            (0, 40),
            (0, 41),
            (0, 42),
            (0, 43),
            (0, 44),
            (0, 45),
            (0, 46),
            (0, 47),
            (0, 48),
            (0, 49),
            (0, 50),
            (0, 51),
            (0, 52),
            (0, 53),
            (0, 54),
            (0, 55),
            (0, 56),
            (0, 57),
            (0, 58),
            (0, 59),
        }:
            return 3
        return 11

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_2_2_outputs, num_attn_1_5_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_6_output, num_attn_1_1_output):
        key = (num_attn_1_6_output, num_attn_1_1_output)
        return 6

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_1_6_outputs, num_attn_1_1_outputs)
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


print(run(["<s>", "0", "3", "3", "3", "1", "</s>"]))
