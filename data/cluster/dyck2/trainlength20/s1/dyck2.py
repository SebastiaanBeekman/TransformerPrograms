import numpy as np
import pandas as pd


def select_closest(keys, queries, predicate):
    scores = [[False for _ in keys] for _ in queries]
    for i, q in enumerate(queries):
        matches = [j for j, k in enumerate(keys[: i + 1]) if predicate(q, k)]
        if not (any(matches)):
            scores[i][0] = True
        else:
            j = min(matches, key=lambda j: len(matches) if j == i else abs(i - j))
            scores[i][j] = True
    return scores


def select(keys, queries, predicate):
    return [[predicate(q, k) for k in keys[: i + 1]] for i, q in enumerate(queries)]


def aggregate(attention, values):
    return [[v for a, v in zip(attn, values) if a][0] for attn in attention]


def aggregate_sum(attention, values):
    return [sum([v for a, v in zip(attn, values) if a]) for attn in attention]


def run(tokens):

    # classifier weights ##########################################
    classifier_weights = pd.read_csv(
        "output/length/rasp/dyck2/trainlength20/s1/dyck2_weights.csv",
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
            return k_position == 38
        elif q_position in {1, 34, 33, 23}:
            return k_position == 20
        elif q_position in {2, 11}:
            return k_position == 10
        elif q_position in {26, 3}:
            return k_position == 26
        elif q_position in {32, 4, 20}:
            return k_position == 25
        elif q_position in {10, 5}:
            return k_position == 9
        elif q_position in {6}:
            return k_position == 4
        elif q_position in {7}:
            return k_position == 6
        elif q_position in {8}:
            return k_position == 7
        elif q_position in {9}:
            return k_position == 8
        elif q_position in {12}:
            return k_position == 11
        elif q_position in {13}:
            return k_position == 12
        elif q_position in {14}:
            return k_position == 13
        elif q_position in {15}:
            return k_position == 14
        elif q_position in {16}:
            return k_position == 15
        elif q_position in {17}:
            return k_position == 16
        elif q_position in {18}:
            return k_position == 17
        elif q_position in {19}:
            return k_position == 18
        elif q_position in {21}:
            return k_position == 22
        elif q_position in {22}:
            return k_position == 21
        elif q_position in {24, 36, 37}:
            return k_position == 27
        elif q_position in {25}:
            return k_position == 28
        elif q_position in {27}:
            return k_position == 29
        elif q_position in {28}:
            return k_position == 36
        elif q_position in {29}:
            return k_position == 33
        elif q_position in {30}:
            return k_position == 35
        elif q_position in {31}:
            return k_position == 34
        elif q_position in {35}:
            return k_position == 24
        elif q_position in {38}:
            return k_position == 31
        elif q_position in {39}:
            return k_position == 39

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 22
        elif q_position in {1}:
            return k_position == 0
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {5}:
            return k_position == 4
        elif q_position in {6}:
            return k_position == 5
        elif q_position in {7}:
            return k_position == 6
        elif q_position in {8}:
            return k_position == 7
        elif q_position in {9}:
            return k_position == 8
        elif q_position in {10}:
            return k_position == 9
        elif q_position in {11}:
            return k_position == 10
        elif q_position in {12}:
            return k_position == 11
        elif q_position in {13}:
            return k_position == 12
        elif q_position in {14}:
            return k_position == 13
        elif q_position in {15}:
            return k_position == 14
        elif q_position in {16}:
            return k_position == 15
        elif q_position in {17}:
            return k_position == 16
        elif q_position in {18}:
            return k_position == 17
        elif q_position in {19, 21}:
            return k_position == 18
        elif q_position in {20, 22, 36}:
            return k_position == 29
        elif q_position in {37, 23}:
            return k_position == 30
        elif q_position in {24, 26}:
            return k_position == 35
        elif q_position in {25}:
            return k_position == 27
        elif q_position in {27, 29, 31}:
            return k_position == 33
        elif q_position in {32, 28}:
            return k_position == 28
        elif q_position in {30}:
            return k_position == 19
        elif q_position in {33}:
            return k_position == 23
        elif q_position in {34}:
            return k_position == 31
        elif q_position in {35, 38}:
            return k_position == 32
        elif q_position in {39}:
            return k_position == 34

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(token, position):
        if token in {"{", "("}:
            return position == 1
        elif token in {")"}:
            return position == 7
        elif token in {"<s>"}:
            return position == 2
        elif token in {"}"}:
            return position == 6

    attn_0_2_pattern = select_closest(positions, tokens, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 22, 23}:
            return k_position == 32
        elif q_position in {1}:
            return k_position == 30
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {3, 20}:
            return k_position == 2
        elif q_position in {33, 27, 4, 30}:
            return k_position == 3
        elif q_position in {5}:
            return k_position == 4
        elif q_position in {6}:
            return k_position == 5
        elif q_position in {7}:
            return k_position == 6
        elif q_position in {8}:
            return k_position == 7
        elif q_position in {9}:
            return k_position == 8
        elif q_position in {10}:
            return k_position == 9
        elif q_position in {11}:
            return k_position == 10
        elif q_position in {12}:
            return k_position == 11
        elif q_position in {13}:
            return k_position == 12
        elif q_position in {14}:
            return k_position == 13
        elif q_position in {15}:
            return k_position == 14
        elif q_position in {16}:
            return k_position == 15
        elif q_position in {17}:
            return k_position == 16
        elif q_position in {18}:
            return k_position == 17
        elif q_position in {19}:
            return k_position == 18
        elif q_position in {21}:
            return k_position == 38
        elif q_position in {24, 37}:
            return k_position == 26
        elif q_position in {25, 26}:
            return k_position == 21
        elif q_position in {28}:
            return k_position == 24
        elif q_position in {35, 29, 39}:
            return k_position == 34
        elif q_position in {32, 31}:
            return k_position == 37
        elif q_position in {34}:
            return k_position == 36
        elif q_position in {36}:
            return k_position == 33
        elif q_position in {38}:
            return k_position == 27

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(token, position):
        if token in {"("}:
            return position == 2
        elif token in {")"}:
            return position == 39
        elif token in {"<s>"}:
            return position == 3
        elif token in {"{"}:
            return position == 5
        elif token in {"}"}:
            return position == 19

    num_attn_0_0_pattern = select(positions, tokens, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_position, k_position):
        if q_position in {0, 31}:
            return k_position == 13
        elif q_position in {1}:
            return k_position == 6
        elif q_position in {2}:
            return k_position == 4
        elif q_position in {9, 3, 6}:
            return k_position == 2
        elif q_position in {34, 4}:
            return k_position == 11
        elif q_position in {5}:
            return k_position == 31
        elif q_position in {26, 7}:
            return k_position == 29
        elif q_position in {8, 35}:
            return k_position == 3
        elif q_position in {10}:
            return k_position == 5
        elif q_position in {11}:
            return k_position == 0
        elif q_position in {25, 12}:
            return k_position == 20
        elif q_position in {16, 13, 14}:
            return k_position == 9
        elif q_position in {15}:
            return k_position == 21
        elif q_position in {17}:
            return k_position == 10
        elif q_position in {18}:
            return k_position == 27
        elif q_position in {19}:
            return k_position == 36
        elif q_position in {20, 21, 38}:
            return k_position == 34
        elif q_position in {22}:
            return k_position == 22
        elif q_position in {27, 23}:
            return k_position == 39
        elif q_position in {24, 37}:
            return k_position == 24
        elif q_position in {33, 28}:
            return k_position == 26
        elif q_position in {29}:
            return k_position == 38
        elif q_position in {30}:
            return k_position == 37
        elif q_position in {32}:
            return k_position == 17
        elif q_position in {36}:
            return k_position == 12
        elif q_position in {39}:
            return k_position == 33

    num_attn_0_1_pattern = select(positions, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(token, position):
        if token in {"("}:
            return position == 21
        elif token in {"}", ")"}:
            return position == 7
        elif token in {"<s>"}:
            return position == 4
        elif token in {"{"}:
            return position == 34

    num_attn_0_2_pattern = select(positions, tokens, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 8
        elif q_position in {1}:
            return k_position == 7
        elif q_position in {2, 27}:
            return k_position == 37
        elif q_position in {19, 3, 37, 30}:
            return k_position == 35
        elif q_position in {16, 4}:
            return k_position == 6
        elif q_position in {8, 5}:
            return k_position == 5
        elif q_position in {12, 6}:
            return k_position == 0
        elif q_position in {7}:
            return k_position == 1
        elif q_position in {9, 34, 38}:
            return k_position == 2
        elif q_position in {17, 10, 14, 25}:
            return k_position == 9
        elif q_position in {11, 15}:
            return k_position == 11
        elif q_position in {13}:
            return k_position == 10
        elif q_position in {18}:
            return k_position == 14
        elif q_position in {20, 29}:
            return k_position == 22
        elif q_position in {33, 21}:
            return k_position == 25
        elif q_position in {22}:
            return k_position == 36
        elif q_position in {31, 23}:
            return k_position == 28
        elif q_position in {24}:
            return k_position == 24
        elif q_position in {32, 26}:
            return k_position == 39
        elif q_position in {28}:
            return k_position == 29
        elif q_position in {35}:
            return k_position == 12
        elif q_position in {36}:
            return k_position == 34
        elif q_position in {39}:
            return k_position == 3

    num_attn_0_3_pattern = select(positions, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_2_output, attn_0_3_output):
        key = (attn_0_2_output, attn_0_3_output)
        if key in {("}", "(")}:
            return 6
        elif key in {("(", ")"), (")", "("), ("<s>", ")"), ("{", ")"), ("{", "}")}:
            return 12
        elif key in {(")", ")"), (")", "}"), ("<s>", "}"), ("}", ")"), ("}", "}")}:
            return 16
        elif key in {("(", "}")}:
            return 10
        elif key in {(")", "<s>"), (")", "{"), ("}", "<s>"), ("}", "{")}:
            return 5
        elif key in {("<s>", "(")}:
            return 7
        return 4

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_0_3_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_3_output, attn_0_0_output):
        key = (attn_0_3_output, attn_0_0_output)
        if key in {
            (")", ")"),
            (")", "}"),
            ("<s>", ")"),
            ("<s>", "}"),
            ("{", ")"),
            ("{", "}"),
            ("}", ")"),
            ("}", "}"),
        }:
            return 31
        return 36

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_3_outputs, attn_0_0_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # mlp_0_2 #####################################################
    def mlp_0_2(token, attn_0_3_output):
        key = (token, attn_0_3_output)
        if key in {(")", "("), (")", "{")}:
            return 21
        elif key in {
            (")", ")"),
            (")", "}"),
            ("<s>", ")"),
            ("<s>", "}"),
            ("}", ")"),
            ("}", "}"),
        }:
            return 16
        elif key in {("}", "{")}:
            return 18
        elif key in {("(", "{"), ("<s>", "{"), ("{", "("), ("{", "{")}:
            return 4
        elif key in {("(", ")"), ("(", "}"), ("{", ")"), ("{", "}")}:
            return 17
        elif key in {(")", "<s>"), ("}", "("), ("}", "<s>")}:
            return 2
        return 1

    mlp_0_2_outputs = [mlp_0_2(k0, k1) for k0, k1 in zip(tokens, attn_0_3_outputs)]
    mlp_0_2_output_scores = classifier_weights.loc[
        [("mlp_0_2_outputs", str(v)) for v in mlp_0_2_outputs]
    ]

    # mlp_0_3 #####################################################
    def mlp_0_3(attn_0_1_output, token):
        key = (attn_0_1_output, token)
        if key in {("{", "}")}:
            return 14
        elif key in {(")", "("), (")", "<s>"), (")", "{")}:
            return 0
        elif key in {("(", "}"), ("<s>", "}"), ("{", ")")}:
            return 5
        elif key in {(")", "}")}:
            return 7
        elif key in {("(", ")"), ("}", "(")}:
            return 27
        return 4

    mlp_0_3_outputs = [mlp_0_3(k0, k1) for k0, k1 in zip(attn_0_1_outputs, tokens)]
    mlp_0_3_output_scores = classifier_weights.loc[
        [("mlp_0_3_outputs", str(v)) for v in mlp_0_3_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_0_output, num_attn_0_2_output):
        key = (num_attn_0_0_output, num_attn_0_2_output)
        return 33

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_1_output, num_attn_0_0_output):
        key = (num_attn_0_1_output, num_attn_0_0_output)
        return 24

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # num_mlp_0_2 #################################################
    def num_mlp_0_2(num_attn_0_2_output, num_attn_0_3_output):
        key = (num_attn_0_2_output, num_attn_0_3_output)
        return 37

    num_mlp_0_2_outputs = [
        num_mlp_0_2(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_0_2_output_scores = classifier_weights.loc[
        [("num_mlp_0_2_outputs", str(v)) for v in num_mlp_0_2_outputs]
    ]

    # num_mlp_0_3 #################################################
    def num_mlp_0_3(num_attn_0_3_output):
        key = num_attn_0_3_output
        return 1

    num_mlp_0_3_outputs = [num_mlp_0_3(k0) for k0 in num_attn_0_3_outputs]
    num_mlp_0_3_output_scores = classifier_weights.loc[
        [("num_mlp_0_3_outputs", str(v)) for v in num_mlp_0_3_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_position, k_position):
        if q_position in {0}:
            return k_position == 30
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2, 3}:
            return k_position == 2
        elif q_position in {
            4,
            5,
            6,
            7,
            8,
            9,
            11,
            13,
            15,
            17,
            19,
            21,
            22,
            23,
            26,
            27,
            29,
            33,
            34,
            35,
            36,
            38,
        }:
            return k_position == 3
        elif q_position in {32, 39, 10, 12, 14, 20, 24, 25, 31}:
            return k_position == 4
        elif q_position in {16, 28, 30}:
            return k_position == 6
        elif q_position in {18}:
            return k_position == 7
        elif q_position in {37}:
            return k_position == 28

    attn_1_0_pattern = select_closest(positions, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, mlp_0_2_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(token, position):
        if token in {"}", ")", "{", "("}:
            return position == 5
        elif token in {"<s>"}:
            return position == 25

    attn_1_1_pattern = select_closest(positions, tokens, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, mlp_0_2_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(attn_0_3_output, position):
        if attn_0_3_output in {"{", "("}:
            return position == 2
        elif attn_0_3_output in {"}", ")"}:
            return position == 5
        elif attn_0_3_output in {"<s>"}:
            return position == 22

    attn_1_2_pattern = select_closest(positions, attn_0_3_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, mlp_0_0_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(position, mlp_0_3_output):
        if position in {0}:
            return mlp_0_3_output == 22
        elif position in {
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
            34,
            37,
        }:
            return mlp_0_3_output == 5
        elif position in {20, 22}:
            return mlp_0_3_output == 21
        elif position in {33, 35, 36, 39, 21, 23, 24, 25, 29, 30, 31}:
            return mlp_0_3_output == 6
        elif position in {26, 28}:
            return mlp_0_3_output == 23
        elif position in {27}:
            return mlp_0_3_output == 30
        elif position in {32}:
            return mlp_0_3_output == 31
        elif position in {38}:
            return mlp_0_3_output == 34

    attn_1_3_pattern = select_closest(mlp_0_3_outputs, positions, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, mlp_0_2_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(mlp_0_1_output, mlp_0_0_output):
        if mlp_0_1_output in {
            0,
            2,
            5,
            6,
            9,
            11,
            13,
            14,
            15,
            17,
            21,
            22,
            25,
            26,
            29,
            30,
            32,
            33,
            35,
        }:
            return mlp_0_0_output == 12
        elif mlp_0_1_output in {1, 3, 4, 36, 37, 7, 38, 39, 19, 20, 23, 24, 27, 28, 31}:
            return mlp_0_0_output == 16
        elif mlp_0_1_output in {34, 8, 12, 16, 18}:
            return mlp_0_0_output == 24
        elif mlp_0_1_output in {10}:
            return mlp_0_0_output == 1

    num_attn_1_0_pattern = select(mlp_0_0_outputs, mlp_0_1_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_2_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(mlp_0_2_output, mlp_0_0_output):
        if mlp_0_2_output in {0, 24, 22, 31}:
            return mlp_0_0_output == 13
        elif mlp_0_2_output in {1, 36}:
            return mlp_0_0_output == 24
        elif mlp_0_2_output in {32, 2, 38, 39, 15}:
            return mlp_0_0_output == 30
        elif mlp_0_2_output in {33, 3, 23}:
            return mlp_0_0_output == 28
        elif mlp_0_2_output in {4, 6}:
            return mlp_0_0_output == 7
        elif mlp_0_2_output in {5}:
            return mlp_0_0_output == 38
        elif mlp_0_2_output in {34, 37, 7, 8, 9, 12, 14, 16, 18, 21, 25, 28, 29, 30}:
            return mlp_0_0_output == 4
        elif mlp_0_2_output in {10}:
            return mlp_0_0_output == 9
        elif mlp_0_2_output in {11}:
            return mlp_0_0_output == 22
        elif mlp_0_2_output in {13}:
            return mlp_0_0_output == 33
        elif mlp_0_2_output in {17, 26}:
            return mlp_0_0_output == 12
        elif mlp_0_2_output in {35, 19}:
            return mlp_0_0_output == 34
        elif mlp_0_2_output in {20}:
            return mlp_0_0_output == 26
        elif mlp_0_2_output in {27}:
            return mlp_0_0_output == 25

    num_attn_1_1_pattern = select(mlp_0_0_outputs, mlp_0_2_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_3_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(q_attn_0_2_output, k_attn_0_2_output):
        if q_attn_0_2_output in {")", "}", "<s>", "{", "("}:
            return k_attn_0_2_output == ""

    num_attn_1_2_pattern = select(attn_0_2_outputs, attn_0_2_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_3_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(attn_0_0_output, mlp_0_0_output):
        if attn_0_0_output in {")", "}", "<s>", "{", "("}:
            return mlp_0_0_output == 6

    num_attn_1_3_pattern = select(mlp_0_0_outputs, attn_0_0_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_0_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_2_output, attn_1_3_output):
        key = (attn_1_2_output, attn_1_3_output)
        if key in {
            (0, 36),
            (1, 36),
            (2, 36),
            (3, 36),
            (5, 36),
            (6, 36),
            (8, 36),
            (9, 36),
            (10, 36),
            (11, 36),
            (13, 5),
            (13, 16),
            (13, 19),
            (13, 23),
            (13, 28),
            (13, 31),
            (13, 36),
            (14, 36),
            (15, 36),
            (16, 36),
            (17, 36),
            (18, 19),
            (18, 24),
            (18, 26),
            (18, 28),
            (18, 36),
            (19, 19),
            (19, 28),
            (19, 31),
            (19, 32),
            (19, 36),
            (20, 36),
            (21, 36),
            (22, 36),
            (23, 36),
            (24, 36),
            (25, 36),
            (26, 36),
            (27, 3),
            (27, 16),
            (27, 19),
            (27, 20),
            (27, 23),
            (27, 24),
            (27, 26),
            (27, 27),
            (27, 28),
            (27, 29),
            (27, 31),
            (27, 32),
            (27, 34),
            (27, 35),
            (27, 36),
            (27, 39),
            (28, 36),
            (29, 36),
            (30, 16),
            (30, 19),
            (30, 23),
            (30, 24),
            (30, 26),
            (30, 28),
            (30, 31),
            (30, 32),
            (30, 36),
            (31, 36),
            (32, 5),
            (32, 16),
            (32, 19),
            (32, 23),
            (32, 26),
            (32, 28),
            (32, 31),
            (32, 36),
            (33, 36),
            (34, 19),
            (34, 23),
            (34, 26),
            (34, 28),
            (34, 36),
            (35, 36),
            (36, 36),
            (37, 36),
            (38, 36),
            (39, 36),
        }:
            return 19
        elif key in {(18, 23)}:
            return 21
        elif key in {(22, 23)}:
            return 32
        return 38

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_2_outputs, attn_1_3_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(position, token):
        key = (position, token)
        if key in {
            (0, ")"),
            (3, ")"),
            (3, "}"),
            (6, ")"),
            (6, "<s>"),
            (6, "{"),
            (6, "}"),
            (7, ")"),
            (7, "<s>"),
            (7, "}"),
            (8, ")"),
            (8, "<s>"),
            (8, "}"),
            (9, ")"),
            (9, "<s>"),
            (9, "}"),
            (11, ")"),
            (12, ")"),
            (13, ")"),
            (14, ")"),
            (15, ")"),
            (17, ")"),
            (17, "<s>"),
            (17, "}"),
            (19, ")"),
            (20, ")"),
            (21, ")"),
            (22, ")"),
            (23, ")"),
            (24, ")"),
            (25, ")"),
            (26, ")"),
            (27, ")"),
            (28, ")"),
            (29, ")"),
            (30, ")"),
            (31, ")"),
            (32, ")"),
            (33, ")"),
            (34, ")"),
            (35, ")"),
            (36, ")"),
            (37, ")"),
            (38, ")"),
            (39, ")"),
        }:
            return 14
        elif key in {
            (0, "("),
            (0, "<s>"),
            (3, "("),
            (3, "<s>"),
            (4, "("),
            (4, ")"),
            (4, "<s>"),
            (6, "("),
        }:
            return 11
        elif key in {
            (1, "("),
            (1, "<s>"),
            (1, "{"),
            (2, "("),
            (2, "{"),
            (18, "("),
            (18, "{"),
        }:
            return 7
        elif key in {(1, ")"), (2, ")"), (18, ")"), (18, "<s>")}:
            return 37
        elif key in {(1, "}")}:
            return 24
        return 30

    mlp_1_1_outputs = [mlp_1_1(k0, k1) for k0, k1 in zip(positions, tokens)]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # mlp_1_2 #####################################################
    def mlp_1_2(position, token):
        key = (position, token)
        if key in {
            (5, "("),
            (5, "{"),
            (7, "("),
            (7, ")"),
            (7, "<s>"),
            (7, "{"),
            (7, "}"),
            (9, "("),
            (9, ")"),
            (9, "<s>"),
            (9, "{"),
            (9, "}"),
            (11, ")"),
            (11, "<s>"),
            (11, "}"),
            (13, ")"),
            (13, "<s>"),
            (13, "}"),
            (15, "("),
            (15, ")"),
            (15, "<s>"),
            (15, "{"),
            (15, "}"),
            (17, "("),
            (17, ")"),
            (17, "<s>"),
            (17, "{"),
            (17, "}"),
        }:
            return 20
        elif key in {
            (6, "("),
            (6, "{"),
            (8, "("),
            (8, "{"),
            (10, "("),
            (10, "{"),
            (11, "("),
            (11, "{"),
            (12, "("),
            (12, "{"),
            (13, "("),
            (13, "{"),
            (14, "("),
            (14, "{"),
            (16, "("),
            (16, "{"),
            (18, "("),
            (18, "{"),
            (19, "("),
            (19, ")"),
            (19, "<s>"),
            (19, "{"),
            (19, "}"),
            (20, "("),
            (20, "{"),
            (21, "("),
            (21, "{"),
            (22, "("),
            (22, "{"),
            (23, "("),
            (23, "{"),
            (24, "("),
            (24, "{"),
            (25, "("),
            (25, "{"),
            (26, "("),
            (26, "{"),
            (27, "("),
            (27, "{"),
            (28, "("),
            (28, "{"),
            (29, "("),
            (29, "{"),
            (30, "("),
            (30, "{"),
            (31, "("),
            (31, "{"),
            (32, "("),
            (32, "{"),
            (33, "("),
            (33, "{"),
            (34, "("),
            (34, "{"),
            (35, "("),
            (35, "{"),
            (36, "("),
            (36, "{"),
            (37, "("),
            (37, "{"),
            (38, "("),
            (38, "{"),
            (39, "("),
            (39, "{"),
        }:
            return 26
        elif key in {
            (0, "("),
            (0, "<s>"),
            (0, "{"),
            (1, "("),
            (1, "<s>"),
            (1, "{"),
            (2, "("),
            (2, "{"),
            (3, "("),
            (3, "<s>"),
            (3, "{"),
            (4, "("),
            (4, "<s>"),
            (4, "{"),
        }:
            return 37
        elif key in {(1, ")"), (1, "}")}:
            return 5
        return 39

    mlp_1_2_outputs = [mlp_1_2(k0, k1) for k0, k1 in zip(positions, tokens)]
    mlp_1_2_output_scores = classifier_weights.loc[
        [("mlp_1_2_outputs", str(v)) for v in mlp_1_2_outputs]
    ]

    # mlp_1_3 #####################################################
    def mlp_1_3(attn_1_2_output, attn_1_1_output):
        key = (attn_1_2_output, attn_1_1_output)
        if key in {
            (0, 2),
            (0, 12),
            (0, 16),
            (0, 19),
            (0, 27),
            (0, 28),
            (1, 2),
            (1, 16),
            (1, 19),
            (1, 27),
            (1, 28),
            (2, 0),
            (2, 2),
            (2, 3),
            (2, 6),
            (2, 7),
            (2, 8),
            (2, 9),
            (2, 10),
            (2, 12),
            (2, 16),
            (2, 19),
            (2, 22),
            (2, 23),
            (2, 24),
            (2, 25),
            (2, 27),
            (2, 28),
            (2, 30),
            (2, 31),
            (2, 33),
            (2, 34),
            (2, 35),
            (2, 37),
            (2, 39),
            (3, 0),
            (3, 2),
            (3, 3),
            (3, 6),
            (3, 9),
            (3, 12),
            (3, 16),
            (3, 19),
            (3, 22),
            (3, 23),
            (3, 24),
            (3, 25),
            (3, 27),
            (3, 28),
            (3, 29),
            (3, 31),
            (3, 33),
            (3, 35),
            (3, 37),
            (3, 39),
            (4, 16),
            (5, 2),
            (5, 3),
            (5, 12),
            (5, 16),
            (5, 19),
            (5, 24),
            (5, 27),
            (5, 28),
            (5, 31),
            (6, 2),
            (6, 12),
            (6, 16),
            (6, 19),
            (7, 0),
            (7, 2),
            (7, 3),
            (7, 12),
            (7, 16),
            (7, 19),
            (7, 24),
            (7, 25),
            (7, 27),
            (7, 28),
            (7, 31),
            (7, 35),
            (8, 7),
            (8, 12),
            (8, 16),
            (8, 19),
            (8, 27),
            (8, 30),
            (9, 2),
            (9, 12),
            (9, 16),
            (9, 19),
            (9, 27),
            (9, 28),
            (10, 16),
            (11, 16),
            (11, 19),
            (12, 0),
            (12, 2),
            (12, 3),
            (12, 4),
            (12, 5),
            (12, 6),
            (12, 7),
            (12, 8),
            (12, 9),
            (12, 10),
            (12, 11),
            (12, 12),
            (12, 13),
            (12, 15),
            (12, 16),
            (12, 17),
            (12, 19),
            (12, 20),
            (12, 22),
            (12, 23),
            (12, 24),
            (12, 25),
            (12, 26),
            (12, 27),
            (12, 28),
            (12, 29),
            (12, 30),
            (12, 31),
            (12, 32),
            (12, 33),
            (12, 34),
            (12, 35),
            (12, 36),
            (12, 37),
            (12, 38),
            (12, 39),
            (13, 12),
            (13, 16),
            (13, 19),
            (14, 16),
            (14, 19),
            (14, 27),
            (15, 16),
            (15, 19),
            (16, 2),
            (16, 3),
            (16, 4),
            (16, 5),
            (16, 7),
            (16, 9),
            (16, 12),
            (16, 16),
            (16, 19),
            (16, 22),
            (16, 23),
            (16, 24),
            (16, 25),
            (16, 26),
            (16, 27),
            (16, 28),
            (16, 33),
            (16, 35),
            (16, 37),
            (16, 39),
            (17, 7),
            (17, 12),
            (17, 16),
            (17, 19),
            (18, 16),
            (18, 19),
            (19, 0),
            (19, 1),
            (19, 2),
            (19, 3),
            (19, 6),
            (19, 7),
            (19, 8),
            (19, 9),
            (19, 10),
            (19, 11),
            (19, 12),
            (19, 13),
            (19, 15),
            (19, 16),
            (19, 19),
            (19, 20),
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
            (20, 16),
            (20, 19),
            (21, 16),
            (21, 19),
            (22, 16),
            (22, 19),
            (23, 7),
            (23, 12),
            (23, 16),
            (23, 19),
            (23, 27),
            (24, 16),
            (24, 19),
            (25, 7),
            (25, 12),
            (25, 16),
            (25, 19),
            (26, 2),
            (26, 12),
            (26, 16),
            (26, 19),
            (26, 27),
            (27, 2),
            (27, 16),
            (27, 19),
            (28, 7),
            (28, 12),
            (28, 16),
            (28, 19),
            (29, 2),
            (29, 16),
            (29, 19),
            (30, 2),
            (30, 12),
            (30, 16),
            (30, 19),
            (30, 24),
            (30, 27),
            (30, 28),
            (30, 30),
            (30, 31),
            (31, 0),
            (31, 2),
            (31, 3),
            (31, 6),
            (31, 7),
            (31, 8),
            (31, 9),
            (31, 12),
            (31, 16),
            (31, 19),
            (31, 24),
            (31, 25),
            (31, 27),
            (31, 28),
            (31, 31),
            (31, 33),
            (31, 35),
            (31, 39),
            (32, 2),
            (32, 12),
            (32, 16),
            (32, 19),
            (32, 27),
            (33, 2),
            (33, 12),
            (33, 16),
            (33, 19),
            (33, 27),
            (34, 7),
            (34, 12),
            (34, 16),
            (34, 19),
            (35, 0),
            (35, 2),
            (35, 3),
            (35, 8),
            (35, 12),
            (35, 16),
            (35, 19),
            (35, 24),
            (35, 25),
            (35, 27),
            (35, 28),
            (35, 31),
            (35, 33),
            (35, 35),
            (35, 39),
            (36, 0),
            (36, 2),
            (36, 3),
            (36, 16),
            (36, 19),
            (36, 24),
            (36, 25),
            (36, 27),
            (36, 28),
            (36, 35),
            (37, 0),
            (37, 2),
            (37, 3),
            (37, 12),
            (37, 16),
            (37, 19),
            (37, 24),
            (37, 25),
            (37, 27),
            (37, 28),
            (37, 31),
            (37, 35),
            (38, 2),
            (38, 12),
            (38, 16),
            (38, 19),
            (38, 24),
            (38, 25),
            (38, 27),
            (38, 28),
            (38, 31),
            (39, 2),
            (39, 12),
            (39, 16),
            (39, 19),
            (39, 27),
            (39, 28),
        }:
            return 38
        elif key in {
            (0, 14),
            (0, 30),
            (1, 12),
            (1, 14),
            (1, 30),
            (2, 14),
            (3, 1),
            (3, 8),
            (3, 10),
            (3, 14),
            (3, 30),
            (3, 34),
            (4, 14),
            (5, 14),
            (5, 30),
            (6, 14),
            (7, 14),
            (7, 30),
            (8, 14),
            (9, 14),
            (9, 30),
            (10, 14),
            (11, 14),
            (12, 1),
            (12, 14),
            (13, 14),
            (14, 12),
            (14, 14),
            (14, 30),
            (15, 14),
            (16, 0),
            (16, 1),
            (16, 6),
            (16, 8),
            (16, 10),
            (16, 14),
            (16, 30),
            (16, 31),
            (16, 38),
            (17, 14),
            (18, 14),
            (19, 14),
            (20, 14),
            (21, 14),
            (22, 14),
            (23, 14),
            (23, 30),
            (24, 14),
            (25, 14),
            (26, 14),
            (26, 30),
            (27, 12),
            (27, 14),
            (27, 30),
            (28, 14),
            (29, 12),
            (29, 14),
            (29, 30),
            (30, 14),
            (31, 14),
            (31, 30),
            (32, 14),
            (32, 30),
            (33, 14),
            (34, 14),
            (35, 14),
            (35, 30),
            (36, 8),
            (36, 12),
            (36, 14),
            (36, 30),
            (36, 31),
            (37, 14),
            (37, 30),
            (38, 14),
            (38, 30),
            (39, 14),
            (39, 30),
        }:
            return 15
        elif key in {
            (0, 7),
            (1, 7),
            (2, 5),
            (3, 5),
            (3, 7),
            (5, 5),
            (5, 7),
            (5, 35),
            (7, 5),
            (7, 7),
            (9, 7),
            (10, 7),
            (14, 7),
            (19, 5),
            (27, 7),
            (30, 7),
            (31, 5),
            (32, 7),
            (33, 7),
            (35, 5),
            (35, 7),
            (36, 5),
            (37, 5),
            (37, 7),
            (38, 7),
            (39, 7),
        }:
            return 6
        elif key in {(21, 7), (24, 7), (26, 7), (29, 7), (36, 7)}:
            return 3
        return 1

    mlp_1_3_outputs = [
        mlp_1_3(k0, k1) for k0, k1 in zip(attn_1_2_outputs, attn_1_1_outputs)
    ]
    mlp_1_3_output_scores = classifier_weights.loc[
        [("mlp_1_3_outputs", str(v)) for v in mlp_1_3_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_1_output, num_attn_0_3_output):
        key = (num_attn_1_1_output, num_attn_0_3_output)
        return 9

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_0_output, num_attn_1_3_output):
        key = (num_attn_1_0_output, num_attn_1_3_output)
        return 17

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_1_3_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # num_mlp_1_2 #################################################
    def num_mlp_1_2(num_attn_1_2_output):
        key = num_attn_1_2_output
        return 15

    num_mlp_1_2_outputs = [num_mlp_1_2(k0) for k0 in num_attn_1_2_outputs]
    num_mlp_1_2_output_scores = classifier_weights.loc[
        [("num_mlp_1_2_outputs", str(v)) for v in num_mlp_1_2_outputs]
    ]

    # num_mlp_1_3 #################################################
    def num_mlp_1_3(num_attn_0_2_output, num_attn_1_0_output):
        key = (num_attn_0_2_output, num_attn_1_0_output)
        if key in {
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
            (0, 60),
            (0, 61),
            (0, 62),
            (0, 63),
            (0, 64),
            (0, 65),
            (0, 66),
            (0, 67),
            (0, 68),
            (0, 69),
            (0, 70),
            (0, 71),
            (0, 72),
            (0, 73),
            (0, 74),
            (0, 75),
            (0, 76),
            (0, 77),
            (0, 78),
            (0, 79),
            (1, 54),
            (1, 55),
            (1, 56),
            (1, 57),
            (1, 58),
            (1, 59),
            (1, 60),
            (1, 61),
            (1, 62),
            (1, 63),
            (1, 64),
            (1, 65),
            (1, 66),
            (1, 67),
            (1, 68),
            (1, 69),
            (1, 70),
            (1, 71),
            (1, 72),
            (1, 73),
            (1, 74),
            (1, 75),
            (1, 76),
            (1, 77),
            (1, 78),
            (1, 79),
            (2, 69),
            (2, 70),
            (2, 71),
            (2, 72),
            (2, 73),
            (2, 74),
            (2, 75),
            (2, 76),
            (2, 77),
            (2, 78),
            (2, 79),
        }:
            return 37
        return 33

    num_mlp_1_3_outputs = [
        num_mlp_1_3(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_1_0_outputs)
    ]
    num_mlp_1_3_output_scores = classifier_weights.loc[
        [("num_mlp_1_3_outputs", str(v)) for v in num_mlp_1_3_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(token, position):
        if token in {"{", "("}:
            return position == 4
        elif token in {"}", "<s>", ")"}:
            return position == 5

    attn_2_0_pattern = select_closest(positions, tokens, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_3_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(token, position):
        if token in {"{", "("}:
            return position == 5
        elif token in {"}", ")"}:
            return position == 4
        elif token in {"<s>"}:
            return position == 6

    attn_2_1_pattern = select_closest(positions, tokens, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_0_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(token, position):
        if token in {"{", "("}:
            return position == 5
        elif token in {"}", "<s>", ")"}:
            return position == 6

    attn_2_2_pattern = select_closest(positions, tokens, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, mlp_0_2_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(attn_0_1_output, mlp_0_0_output):
        if attn_0_1_output in {"}", "{", "("}:
            return mlp_0_0_output == 6
        elif attn_0_1_output in {"<s>", ")"}:
            return mlp_0_0_output == 5

    attn_2_3_pattern = select_closest(mlp_0_0_outputs, attn_0_1_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_3_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(position, mlp_0_2_output):
        if position in {0, 35, 36, 38, 27, 30, 31}:
            return mlp_0_2_output == 19
        elif position in {32, 1, 2, 37, 39, 17, 20, 21, 22, 24, 25}:
            return mlp_0_2_output == 6
        elif position in {34, 3, 6, 7, 9, 11, 13, 15, 19, 23, 26}:
            return mlp_0_2_output == 2
        elif position in {33, 4, 8, 10, 12, 14, 16, 18, 28, 29}:
            return mlp_0_2_output == 16
        elif position in {5}:
            return mlp_0_2_output == 21

    num_attn_2_0_pattern = select(mlp_0_2_outputs, positions, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_3_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_0_2_output, mlp_0_3_output):
        if attn_0_2_output in {"<s>", ")", "{", "("}:
            return mlp_0_3_output == 27
        elif attn_0_2_output in {"}"}:
            return mlp_0_3_output == 14

    num_attn_2_1_pattern = select(mlp_0_3_outputs, attn_0_2_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_2_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(mlp_0_2_output, attn_1_2_output):
        if mlp_0_2_output in {0, 32, 15, 22, 29}:
            return attn_1_2_output == 12
        elif mlp_0_2_output in {1}:
            return attn_1_2_output == 26
        elif mlp_0_2_output in {
            2,
            3,
            5,
            7,
            8,
            9,
            11,
            13,
            14,
            16,
            17,
            18,
            19,
            20,
            21,
            23,
            24,
            25,
            26,
            27,
            28,
            30,
            31,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
        }:
            return attn_1_2_output == 4
        elif mlp_0_2_output in {4}:
            return attn_1_2_output == 39
        elif mlp_0_2_output in {12, 6}:
            return attn_1_2_output == 28
        elif mlp_0_2_output in {10}:
            return attn_1_2_output == 27

    num_attn_2_2_pattern = select(attn_1_2_outputs, mlp_0_2_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_2_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_1_0_output, mlp_0_3_output):
        if attn_1_0_output in {
            0,
            1,
            2,
            3,
            4,
            6,
            7,
            9,
            11,
            15,
            20,
            21,
            23,
            25,
            29,
            34,
            36,
            38,
            39,
        }:
            return mlp_0_3_output == 5
        elif attn_1_0_output in {32, 5, 10, 13, 22, 24}:
            return mlp_0_3_output == 16
        elif attn_1_0_output in {33, 35, 37, 8, 12, 14, 16, 17, 18, 19, 26, 27, 28, 30}:
            return mlp_0_3_output == 7
        elif attn_1_0_output in {31}:
            return mlp_0_3_output == 6

    num_attn_2_3_pattern = select(mlp_0_3_outputs, attn_1_0_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, ones)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_1_output, attn_2_0_output):
        key = (attn_2_1_output, attn_2_0_output)
        if key in {
            (0, 3),
            (0, 14),
            (0, 18),
            (0, 38),
            (1, 0),
            (1, 2),
            (1, 5),
            (1, 8),
            (1, 9),
            (1, 10),
            (1, 12),
            (1, 13),
            (1, 15),
            (1, 16),
            (1, 17),
            (1, 19),
            (1, 20),
            (1, 21),
            (1, 22),
            (1, 25),
            (1, 28),
            (1, 29),
            (1, 36),
            (1, 39),
            (2, 14),
            (3, 14),
            (3, 18),
            (3, 28),
            (3, 34),
            (3, 36),
            (3, 38),
            (5, 14),
            (6, 0),
            (6, 3),
            (6, 14),
            (6, 17),
            (6, 18),
            (6, 23),
            (6, 24),
            (6, 26),
            (6, 27),
            (6, 28),
            (6, 34),
            (6, 36),
            (6, 37),
            (6, 38),
            (7, 14),
            (8, 14),
            (8, 18),
            (9, 3),
            (9, 14),
            (9, 18),
            (9, 23),
            (9, 28),
            (9, 38),
            (10, 14),
            (10, 18),
            (11, 3),
            (11, 14),
            (11, 18),
            (11, 23),
            (11, 38),
            (12, 3),
            (12, 4),
            (12, 14),
            (12, 17),
            (12, 18),
            (12, 23),
            (12, 24),
            (12, 28),
            (12, 34),
            (12, 35),
            (12, 37),
            (12, 38),
            (13, 14),
            (13, 18),
            (14, 16),
            (14, 17),
            (15, 14),
            (15, 18),
            (17, 0),
            (17, 2),
            (17, 3),
            (17, 5),
            (17, 6),
            (17, 8),
            (17, 9),
            (17, 10),
            (17, 11),
            (17, 12),
            (17, 13),
            (17, 14),
            (17, 15),
            (17, 16),
            (17, 17),
            (17, 18),
            (17, 19),
            (17, 20),
            (17, 21),
            (17, 22),
            (17, 23),
            (17, 24),
            (17, 25),
            (17, 26),
            (17, 27),
            (17, 28),
            (17, 29),
            (17, 30),
            (17, 31),
            (17, 32),
            (17, 33),
            (17, 34),
            (17, 35),
            (17, 36),
            (17, 37),
            (17, 38),
            (17, 39),
            (19, 3),
            (19, 14),
            (19, 18),
            (19, 38),
            (20, 14),
            (20, 18),
            (22, 14),
            (22, 18),
            (23, 0),
            (23, 3),
            (23, 9),
            (23, 14),
            (23, 17),
            (23, 18),
            (23, 23),
            (23, 24),
            (23, 26),
            (23, 27),
            (23, 28),
            (23, 29),
            (23, 31),
            (23, 32),
            (23, 33),
            (23, 34),
            (23, 35),
            (23, 36),
            (23, 37),
            (23, 38),
            (23, 39),
            (24, 0),
            (24, 3),
            (24, 6),
            (24, 14),
            (24, 17),
            (24, 18),
            (24, 23),
            (24, 24),
            (24, 26),
            (24, 27),
            (24, 28),
            (24, 30),
            (24, 31),
            (24, 33),
            (24, 34),
            (24, 35),
            (24, 36),
            (24, 37),
            (24, 38),
            (24, 39),
            (25, 14),
            (25, 18),
            (26, 3),
            (26, 14),
            (26, 17),
            (26, 18),
            (26, 23),
            (26, 26),
            (26, 28),
            (26, 34),
            (26, 36),
            (26, 38),
            (27, 0),
            (27, 3),
            (27, 9),
            (27, 13),
            (27, 14),
            (27, 17),
            (27, 18),
            (27, 23),
            (27, 24),
            (27, 26),
            (27, 27),
            (27, 28),
            (27, 29),
            (27, 30),
            (27, 31),
            (27, 32),
            (27, 33),
            (27, 34),
            (27, 35),
            (27, 36),
            (27, 37),
            (27, 38),
            (27, 39),
            (28, 3),
            (28, 6),
            (28, 14),
            (28, 17),
            (28, 18),
            (28, 23),
            (28, 28),
            (28, 34),
            (28, 36),
            (28, 38),
            (29, 3),
            (29, 14),
            (29, 18),
            (29, 38),
            (30, 0),
            (30, 3),
            (30, 14),
            (30, 17),
            (30, 18),
            (30, 24),
            (30, 26),
            (30, 27),
            (30, 28),
            (30, 34),
            (30, 35),
            (30, 36),
            (30, 37),
            (30, 38),
            (31, 3),
            (31, 14),
            (31, 18),
            (31, 23),
            (31, 28),
            (31, 38),
            (32, 14),
            (32, 17),
            (32, 23),
            (32, 26),
            (32, 28),
            (32, 34),
            (32, 36),
            (32, 38),
            (33, 3),
            (33, 14),
            (33, 23),
            (33, 38),
            (34, 0),
            (34, 3),
            (34, 14),
            (34, 17),
            (34, 18),
            (34, 23),
            (34, 27),
            (34, 28),
            (34, 33),
            (34, 34),
            (34, 35),
            (34, 36),
            (34, 39),
            (35, 3),
            (35, 14),
            (35, 18),
            (35, 23),
            (35, 28),
            (35, 34),
            (35, 38),
            (36, 0),
            (36, 17),
            (36, 23),
            (36, 26),
            (36, 28),
            (36, 34),
            (36, 36),
            (36, 37),
            (37, 0),
            (37, 3),
            (37, 14),
            (37, 18),
            (37, 23),
            (37, 26),
            (37, 27),
            (37, 28),
            (37, 34),
            (37, 36),
            (37, 38),
            (38, 3),
            (38, 14),
            (38, 18),
            (38, 23),
            (39, 3),
            (39, 14),
            (39, 18),
            (39, 23),
            (39, 34),
            (39, 38),
        }:
            return 39
        elif key in {
            (0, 4),
            (1, 1),
            (1, 3),
            (1, 4),
            (1, 6),
            (1, 7),
            (1, 11),
            (1, 14),
            (1, 18),
            (1, 23),
            (1, 24),
            (1, 26),
            (1, 27),
            (1, 30),
            (1, 31),
            (1, 32),
            (1, 33),
            (1, 34),
            (1, 35),
            (1, 37),
            (1, 38),
            (3, 3),
            (3, 4),
            (3, 6),
            (3, 23),
            (5, 4),
            (6, 1),
            (6, 4),
            (6, 6),
            (7, 4),
            (8, 1),
            (8, 4),
            (9, 4),
            (10, 4),
            (11, 4),
            (14, 1),
            (14, 3),
            (14, 4),
            (14, 6),
            (14, 14),
            (14, 23),
            (14, 24),
            (14, 27),
            (14, 30),
            (14, 31),
            (14, 32),
            (14, 33),
            (14, 34),
            (14, 35),
            (14, 37),
            (15, 4),
            (17, 1),
            (17, 4),
            (20, 4),
            (22, 4),
            (23, 1),
            (23, 4),
            (25, 4),
            (26, 4),
            (29, 4),
            (30, 1),
            (30, 4),
            (30, 6),
            (30, 23),
            (31, 4),
            (32, 4),
            (32, 6),
            (33, 4),
            (35, 4),
            (36, 4),
            (36, 6),
            (36, 14),
            (36, 30),
            (37, 4),
            (38, 4),
            (39, 4),
        }:
            return 27
        elif key in {
            (4, 0),
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 4),
            (4, 5),
            (4, 6),
            (4, 8),
            (4, 9),
            (4, 10),
            (4, 11),
            (4, 12),
            (4, 13),
            (4, 14),
            (4, 15),
            (4, 16),
            (4, 17),
            (4, 18),
            (4, 19),
            (4, 20),
            (4, 21),
            (4, 22),
            (4, 23),
            (4, 24),
            (4, 25),
            (4, 26),
            (4, 27),
            (4, 28),
            (4, 29),
            (4, 30),
            (4, 31),
            (4, 32),
            (4, 33),
            (4, 34),
            (4, 35),
            (4, 36),
            (4, 37),
            (4, 38),
            (4, 39),
            (13, 4),
            (19, 4),
            (24, 4),
            (27, 4),
            (28, 4),
            (34, 4),
        }:
            return 7
        elif key in {
            (26, 27),
            (32, 3),
            (34, 24),
            (34, 26),
            (34, 31),
            (34, 37),
            (34, 38),
            (36, 3),
            (36, 24),
            (36, 27),
            (36, 38),
        }:
            return 0
        elif key in {
            (3, 30),
            (4, 7),
            (6, 30),
            (14, 0),
            (14, 7),
            (23, 6),
            (26, 6),
            (27, 6),
            (30, 30),
            (34, 6),
        }:
            return 6
        elif key in {(32, 27)}:
            return 30
        return 21

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_1_outputs, attn_2_0_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_2_1_output, token):
        key = (attn_2_1_output, token)
        if key in {
            (0, "("),
            (3, "("),
            (7, "("),
            (7, "{"),
            (8, "("),
            (8, "<s>"),
            (8, "{"),
            (9, "("),
            (11, "("),
            (12, "("),
            (13, "("),
            (13, "<s>"),
            (13, "{"),
            (14, "("),
            (14, "<s>"),
            (14, "{"),
            (17, "("),
            (17, "<s>"),
            (17, "{"),
            (18, "("),
            (18, "<s>"),
            (18, "{"),
            (19, "("),
            (22, "("),
            (22, "<s>"),
            (22, "{"),
            (23, "("),
            (23, "<s>"),
            (23, "{"),
            (24, "("),
            (24, "<s>"),
            (24, "{"),
            (26, "("),
            (27, "("),
            (27, "{"),
            (30, "("),
            (31, "("),
            (33, "("),
            (35, "("),
            (37, "("),
            (37, "<s>"),
            (37, "{"),
            (38, "("),
            (39, "("),
        }:
            return 31
        elif key in {
            (0, "{"),
            (1, "{"),
            (2, "{"),
            (3, "{"),
            (4, "("),
            (4, ")"),
            (4, "<s>"),
            (4, "{"),
            (4, "}"),
            (5, "{"),
            (6, "{"),
            (9, "{"),
            (10, "{"),
            (11, "{"),
            (12, "{"),
            (15, "("),
            (15, "{"),
            (19, "{"),
            (20, "{"),
            (25, "{"),
            (26, "{"),
            (28, "{"),
            (29, "{"),
            (30, "{"),
            (31, "{"),
            (32, "{"),
            (33, "{"),
            (34, "{"),
            (35, "{"),
            (36, "{"),
            (38, "{"),
            (39, "{"),
        }:
            return 38
        elif key in {(1, "("), (1, "<s>")}:
            return 25
        elif key in {(10, "(")}:
            return 20
        return 22

    mlp_2_1_outputs = [mlp_2_1(k0, k1) for k0, k1 in zip(attn_2_1_outputs, tokens)]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # mlp_2_2 #####################################################
    def mlp_2_2(position, attn_2_1_output):
        key = (position, attn_2_1_output)
        if key in {
            (0, 16),
            (0, 20),
            (0, 26),
            (1, 16),
            (1, 28),
            (1, 34),
            (2, 3),
            (2, 15),
            (2, 16),
            (2, 30),
            (2, 33),
            (2, 34),
            (3, 16),
            (3, 18),
            (3, 26),
            (4, 3),
            (4, 9),
            (4, 15),
            (4, 16),
            (4, 18),
            (4, 20),
            (4, 21),
            (4, 23),
            (4, 24),
            (4, 26),
            (4, 28),
            (4, 30),
            (4, 31),
            (4, 33),
            (4, 34),
            (4, 35),
            (4, 38),
            (6, 16),
            (8, 3),
            (8, 5),
            (8, 6),
            (8, 7),
            (8, 9),
            (8, 10),
            (8, 11),
            (8, 12),
            (8, 13),
            (8, 14),
            (8, 15),
            (8, 16),
            (8, 18),
            (8, 20),
            (8, 21),
            (8, 22),
            (8, 23),
            (8, 24),
            (8, 25),
            (8, 26),
            (8, 28),
            (8, 30),
            (8, 31),
            (8, 32),
            (8, 33),
            (8, 34),
            (8, 35),
            (8, 36),
            (8, 38),
            (14, 0),
            (14, 3),
            (14, 5),
            (14, 7),
            (14, 10),
            (14, 12),
            (14, 13),
            (14, 15),
            (14, 16),
            (14, 18),
            (14, 19),
            (14, 20),
            (14, 21),
            (14, 22),
            (14, 23),
            (14, 24),
            (14, 25),
            (14, 26),
            (14, 27),
            (14, 28),
            (14, 30),
            (14, 32),
            (14, 33),
            (14, 34),
            (14, 35),
            (14, 36),
            (14, 38),
            (16, 0),
            (16, 2),
            (16, 3),
            (16, 5),
            (16, 6),
            (16, 7),
            (16, 8),
            (16, 9),
            (16, 10),
            (16, 11),
            (16, 12),
            (16, 13),
            (16, 14),
            (16, 15),
            (16, 16),
            (16, 17),
            (16, 18),
            (16, 19),
            (16, 20),
            (16, 21),
            (16, 22),
            (16, 23),
            (16, 24),
            (16, 25),
            (16, 26),
            (16, 27),
            (16, 28),
            (16, 29),
            (16, 30),
            (16, 31),
            (16, 32),
            (16, 33),
            (16, 34),
            (16, 35),
            (16, 36),
            (16, 37),
            (16, 38),
            (16, 39),
            (18, 0),
            (18, 3),
            (18, 7),
            (18, 10),
            (18, 12),
            (18, 13),
            (18, 15),
            (18, 16),
            (18, 18),
            (18, 19),
            (18, 21),
            (18, 22),
            (18, 23),
            (18, 24),
            (18, 25),
            (18, 26),
            (18, 27),
            (18, 28),
            (18, 30),
            (18, 32),
            (18, 33),
            (18, 34),
            (18, 35),
            (18, 36),
            (18, 38),
            (19, 17),
            (20, 16),
            (21, 16),
            (21, 33),
            (22, 3),
            (22, 16),
            (22, 30),
            (23, 3),
            (23, 15),
            (23, 16),
            (23, 30),
            (23, 33),
            (24, 3),
            (24, 16),
            (24, 18),
            (24, 23),
            (24, 26),
            (24, 30),
            (24, 33),
            (24, 34),
            (25, 16),
            (25, 26),
            (25, 28),
            (25, 30),
            (25, 33),
            (25, 38),
            (26, 3),
            (26, 12),
            (26, 15),
            (26, 16),
            (26, 23),
            (26, 26),
            (26, 30),
            (26, 33),
            (26, 34),
            (26, 35),
            (27, 16),
            (27, 20),
            (27, 26),
            (27, 28),
            (27, 33),
            (28, 3),
            (28, 12),
            (28, 15),
            (28, 16),
            (28, 23),
            (28, 26),
            (28, 33),
            (28, 34),
            (28, 35),
            (29, 16),
            (30, 16),
            (30, 26),
            (31, 3),
            (31, 16),
            (31, 23),
            (31, 24),
            (31, 26),
            (31, 28),
            (31, 30),
            (31, 33),
            (31, 34),
            (31, 35),
            (31, 38),
            (32, 16),
            (33, 3),
            (33, 16),
            (33, 23),
            (33, 24),
            (33, 26),
            (33, 28),
            (33, 30),
            (33, 33),
            (33, 34),
            (33, 35),
            (33, 38),
            (34, 3),
            (34, 16),
            (34, 23),
            (34, 26),
            (34, 28),
            (34, 30),
            (34, 33),
            (34, 34),
            (34, 35),
            (34, 38),
            (35, 16),
            (35, 26),
            (35, 28),
            (35, 30),
            (35, 38),
            (36, 16),
            (36, 26),
            (37, 16),
            (37, 26),
            (37, 30),
            (37, 38),
            (38, 16),
            (39, 16),
            (39, 26),
            (39, 28),
            (39, 33),
        }:
            return 22
        elif key in {
            (1, 8),
            (3, 6),
            (3, 7),
            (3, 8),
            (3, 9),
            (3, 10),
            (3, 11),
            (3, 13),
            (3, 14),
            (3, 15),
            (3, 22),
            (3, 24),
            (3, 25),
            (3, 28),
            (3, 29),
            (3, 32),
            (3, 34),
            (3, 35),
            (3, 37),
            (3, 38),
            (3, 39),
            (6, 8),
            (6, 22),
            (6, 25),
            (6, 37),
            (6, 38),
            (7, 6),
            (7, 7),
            (7, 8),
            (7, 9),
            (7, 10),
            (7, 11),
            (7, 13),
            (7, 14),
            (7, 15),
            (7, 20),
            (7, 21),
            (7, 22),
            (7, 23),
            (7, 24),
            (7, 25),
            (7, 28),
            (7, 29),
            (7, 30),
            (7, 32),
            (7, 33),
            (7, 34),
            (7, 35),
            (7, 36),
            (7, 37),
            (7, 38),
            (7, 39),
            (11, 6),
            (11, 7),
            (11, 8),
            (11, 9),
            (11, 10),
            (11, 11),
            (11, 12),
            (11, 13),
            (11, 14),
            (11, 15),
            (11, 17),
            (11, 19),
            (11, 21),
            (11, 22),
            (11, 24),
            (11, 25),
            (11, 28),
            (11, 29),
            (11, 32),
            (11, 34),
            (11, 35),
            (11, 37),
            (11, 38),
            (11, 39),
            (13, 6),
            (13, 7),
            (13, 8),
            (13, 9),
            (13, 10),
            (13, 11),
            (13, 12),
            (13, 13),
            (13, 14),
            (13, 15),
            (13, 17),
            (13, 19),
            (13, 20),
            (13, 21),
            (13, 22),
            (13, 23),
            (13, 24),
            (13, 25),
            (13, 27),
            (13, 28),
            (13, 29),
            (13, 32),
            (13, 33),
            (13, 34),
            (13, 35),
            (13, 36),
            (13, 37),
            (13, 38),
            (13, 39),
            (15, 6),
            (15, 7),
            (15, 8),
            (15, 9),
            (15, 10),
            (15, 11),
            (15, 12),
            (15, 13),
            (15, 14),
            (15, 15),
            (15, 17),
            (15, 19),
            (15, 20),
            (15, 21),
            (15, 22),
            (15, 23),
            (15, 24),
            (15, 25),
            (15, 27),
            (15, 28),
            (15, 29),
            (15, 30),
            (15, 32),
            (15, 33),
            (15, 34),
            (15, 35),
            (15, 36),
            (15, 37),
            (15, 38),
            (15, 39),
            (17, 6),
            (17, 7),
            (17, 8),
            (17, 9),
            (17, 10),
            (17, 11),
            (17, 12),
            (17, 13),
            (17, 14),
            (17, 15),
            (17, 17),
            (17, 19),
            (17, 21),
            (17, 22),
            (17, 24),
            (17, 25),
            (17, 28),
            (17, 29),
            (17, 32),
            (17, 34),
            (17, 35),
            (17, 37),
            (17, 38),
            (17, 39),
            (19, 8),
            (20, 8),
            (20, 22),
            (21, 8),
            (21, 37),
            (21, 38),
            (22, 7),
            (22, 8),
            (22, 10),
            (22, 22),
            (22, 25),
            (22, 29),
            (22, 37),
            (22, 38),
            (23, 8),
            (24, 7),
            (24, 8),
            (24, 22),
            (24, 25),
            (24, 29),
            (24, 37),
            (24, 38),
            (26, 8),
            (26, 37),
            (26, 38),
            (28, 8),
            (28, 22),
            (28, 25),
            (28, 37),
            (28, 38),
            (29, 8),
            (29, 38),
            (30, 8),
            (32, 8),
            (36, 8),
            (38, 8),
            (39, 8),
        }:
            return 3
        elif key in {
            (1, 3),
            (1, 6),
            (1, 9),
            (1, 11),
            (1, 13),
            (1, 15),
            (1, 20),
            (1, 23),
            (1, 25),
            (1, 30),
            (1, 31),
            (1, 32),
            (1, 33),
            (1, 35),
            (1, 37),
            (1, 38),
            (1, 39),
            (2, 1),
            (2, 4),
            (2, 29),
            (8, 4),
            (9, 0),
            (9, 1),
            (9, 2),
            (9, 3),
            (9, 4),
            (9, 5),
            (9, 6),
            (9, 7),
            (9, 8),
            (9, 9),
            (9, 10),
            (9, 11),
            (9, 12),
            (9, 13),
            (9, 15),
            (9, 16),
            (9, 18),
            (9, 19),
            (9, 20),
            (9, 21),
            (9, 22),
            (9, 23),
            (9, 24),
            (9, 25),
            (9, 26),
            (9, 27),
            (9, 28),
            (9, 29),
            (9, 30),
            (9, 31),
            (9, 32),
            (9, 33),
            (9, 34),
            (9, 35),
            (9, 36),
            (9, 37),
            (9, 38),
            (9, 39),
            (11, 0),
            (11, 1),
            (11, 2),
            (11, 3),
            (11, 4),
            (11, 5),
            (11, 16),
            (11, 18),
            (11, 20),
            (11, 23),
            (11, 26),
            (11, 27),
            (11, 30),
            (11, 31),
            (11, 33),
            (11, 36),
            (13, 0),
            (13, 1),
            (13, 2),
            (13, 3),
            (13, 4),
            (13, 5),
            (13, 26),
            (13, 30),
            (13, 31),
            (14, 1),
            (14, 4),
            (14, 29),
            (14, 37),
            (15, 0),
            (15, 1),
            (15, 2),
            (15, 3),
            (15, 4),
            (15, 5),
            (15, 16),
            (15, 18),
            (15, 26),
            (15, 31),
            (17, 0),
            (17, 1),
            (17, 3),
            (17, 4),
            (17, 5),
            (17, 16),
            (17, 18),
            (17, 20),
            (17, 23),
            (17, 26),
            (17, 27),
            (17, 30),
            (17, 31),
            (17, 33),
            (17, 36),
            (18, 1),
            (18, 4),
            (18, 29),
            (19, 4),
            (19, 11),
            (19, 37),
            (19, 39),
            (20, 3),
            (20, 6),
            (20, 9),
            (20, 11),
            (20, 13),
            (20, 20),
            (20, 30),
            (20, 32),
            (20, 33),
            (20, 37),
            (20, 38),
            (20, 39),
            (21, 1),
            (21, 4),
            (23, 1),
            (23, 4),
            (23, 29),
            (26, 1),
            (26, 4),
            (28, 1),
            (28, 4),
            (28, 29),
            (28, 30),
            (30, 20),
            (30, 38),
            (32, 3),
            (32, 9),
            (32, 20),
            (32, 30),
            (32, 37),
            (32, 38),
            (36, 20),
            (36, 30),
            (36, 37),
            (36, 38),
            (38, 30),
            (38, 37),
            (38, 38),
            (39, 20),
            (39, 37),
        }:
            return 36
        elif key in {
            (0, 1),
            (0, 4),
            (0, 29),
            (1, 0),
            (1, 1),
            (1, 4),
            (1, 5),
            (1, 7),
            (1, 10),
            (1, 27),
            (1, 29),
            (3, 4),
            (4, 4),
            (5, 1),
            (5, 4),
            (6, 0),
            (6, 1),
            (6, 4),
            (6, 5),
            (6, 6),
            (6, 7),
            (6, 9),
            (6, 10),
            (6, 11),
            (6, 13),
            (6, 15),
            (6, 20),
            (6, 27),
            (6, 29),
            (6, 30),
            (6, 32),
            (6, 39),
            (7, 0),
            (7, 1),
            (7, 3),
            (7, 4),
            (7, 5),
            (7, 27),
            (10, 1),
            (10, 4),
            (10, 29),
            (12, 1),
            (12, 4),
            (12, 29),
            (17, 2),
            (20, 1),
            (20, 4),
            (20, 7),
            (20, 10),
            (20, 27),
            (20, 29),
            (22, 1),
            (22, 4),
            (24, 1),
            (24, 4),
            (25, 1),
            (25, 4),
            (25, 29),
            (27, 1),
            (27, 4),
            (29, 1),
            (29, 4),
            (29, 20),
            (29, 27),
            (29, 29),
            (29, 30),
            (29, 37),
            (30, 0),
            (30, 1),
            (30, 4),
            (30, 5),
            (30, 6),
            (30, 7),
            (30, 9),
            (30, 10),
            (30, 11),
            (30, 13),
            (30, 15),
            (30, 25),
            (30, 27),
            (30, 29),
            (30, 32),
            (30, 37),
            (30, 39),
            (31, 1),
            (31, 4),
            (31, 29),
            (32, 1),
            (32, 4),
            (32, 7),
            (32, 27),
            (32, 29),
            (32, 32),
            (33, 1),
            (33, 4),
            (33, 29),
            (34, 1),
            (34, 4),
            (34, 29),
            (35, 1),
            (35, 4),
            (35, 29),
            (36, 1),
            (36, 4),
            (36, 11),
            (36, 27),
            (36, 29),
            (36, 32),
            (37, 1),
            (37, 4),
            (38, 1),
            (38, 4),
            (38, 7),
            (38, 27),
            (38, 29),
            (39, 1),
            (39, 4),
            (39, 29),
        }:
            return 14
        elif key in {
            (0, 30),
            (0, 38),
            (3, 1),
            (3, 3),
            (3, 20),
            (3, 30),
            (3, 33),
            (4, 1),
            (8, 1),
            (8, 29),
            (16, 1),
            (19, 0),
            (19, 1),
            (19, 2),
            (19, 3),
            (19, 5),
            (19, 6),
            (19, 7),
            (19, 9),
            (19, 10),
            (19, 12),
            (19, 13),
            (19, 14),
            (19, 15),
            (19, 16),
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
            (19, 38),
            (27, 7),
            (27, 30),
            (27, 38),
            (28, 7),
            (30, 30),
            (36, 3),
            (39, 30),
            (39, 38),
        }:
            return 13
        elif key in {
            (2, 37),
            (7, 16),
            (8, 37),
            (9, 14),
            (9, 17),
            (13, 16),
            (13, 18),
            (18, 37),
            (20, 15),
            (20, 23),
            (20, 31),
            (20, 35),
            (21, 29),
            (22, 20),
            (23, 9),
            (23, 11),
            (23, 20),
            (23, 37),
            (25, 37),
            (28, 5),
            (28, 6),
            (28, 9),
            (28, 11),
            (28, 20),
            (28, 32),
            (28, 36),
            (28, 39),
            (31, 37),
            (32, 23),
            (32, 35),
            (33, 37),
            (34, 37),
            (35, 20),
            (35, 37),
            (38, 20),
        }:
            return 9
        elif key in {(2, 38)}:
            return 10
        elif key in {(16, 4)}:
            return 17
        return 8

    mlp_2_2_outputs = [mlp_2_2(k0, k1) for k0, k1 in zip(positions, attn_2_1_outputs)]
    mlp_2_2_output_scores = classifier_weights.loc[
        [("mlp_2_2_outputs", str(v)) for v in mlp_2_2_outputs]
    ]

    # mlp_2_3 #####################################################
    def mlp_2_3(attn_2_1_output, attn_2_3_output):
        key = (attn_2_1_output, attn_2_3_output)
        if key in {
            (0, 1),
            (0, 4),
            (0, 17),
            (1, 1),
            (1, 4),
            (1, 17),
            (1, 23),
            (1, 26),
            (1, 28),
            (1, 30),
            (1, 32),
            (3, 1),
            (3, 4),
            (3, 17),
            (3, 23),
            (3, 30),
            (4, 1),
            (4, 4),
            (4, 14),
            (4, 17),
            (4, 18),
            (4, 23),
            (4, 30),
            (4, 32),
            (5, 1),
            (5, 4),
            (5, 17),
            (6, 1),
            (6, 4),
            (6, 17),
            (7, 4),
            (8, 4),
            (9, 1),
            (9, 4),
            (9, 17),
            (10, 4),
            (11, 1),
            (11, 4),
            (11, 17),
            (14, 1),
            (14, 4),
            (14, 17),
            (15, 4),
            (15, 17),
            (17, 1),
            (17, 3),
            (17, 4),
            (17, 6),
            (17, 11),
            (17, 13),
            (17, 14),
            (17, 15),
            (17, 17),
            (17, 18),
            (17, 23),
            (17, 24),
            (17, 25),
            (17, 26),
            (17, 27),
            (17, 28),
            (17, 29),
            (17, 30),
            (17, 32),
            (17, 34),
            (17, 35),
            (17, 38),
            (17, 39),
            (23, 1),
            (23, 4),
            (23, 17),
            (23, 23),
            (23, 30),
            (23, 32),
            (25, 1),
            (25, 3),
            (25, 4),
            (25, 13),
            (25, 14),
            (25, 17),
            (25, 23),
            (25, 26),
            (25, 28),
            (25, 30),
            (25, 32),
            (25, 35),
            (25, 38),
            (26, 1),
            (26, 4),
            (26, 17),
            (26, 23),
            (27, 1),
            (27, 4),
            (29, 1),
            (29, 4),
            (30, 4),
            (31, 4),
            (32, 1),
            (32, 4),
            (32, 17),
            (32, 23),
            (32, 30),
            (32, 32),
            (33, 1),
            (33, 4),
            (33, 17),
            (34, 1),
            (34, 4),
            (34, 17),
            (35, 1),
            (35, 4),
            (35, 17),
            (37, 1),
            (37, 4),
            (37, 17),
            (38, 1),
            (38, 4),
            (38, 17),
            (39, 1),
            (39, 4),
            (39, 17),
        }:
            return 3
        elif key in {
            (15, 23),
            (15, 30),
            (15, 32),
            (33, 23),
            (35, 23),
            (36, 3),
            (36, 13),
            (36, 14),
            (36, 23),
            (36, 27),
            (36, 28),
            (36, 29),
            (36, 30),
            (36, 32),
            (36, 35),
            (36, 36),
            (36, 38),
            (36, 39),
        }:
            return 31
        elif key in {(36, 1), (36, 4), (39, 23)}:
            return 12
        elif key in {(27, 23)}:
            return 8
        return 17

    mlp_2_3_outputs = [
        mlp_2_3(k0, k1) for k0, k1 in zip(attn_2_1_outputs, attn_2_3_outputs)
    ]
    mlp_2_3_output_scores = classifier_weights.loc[
        [("mlp_2_3_outputs", str(v)) for v in mlp_2_3_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_2_output, num_attn_1_1_output):
        key = (num_attn_1_2_output, num_attn_1_1_output)
        return 34

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_1_1_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_0_output, num_attn_1_0_output):
        key = (num_attn_2_0_output, num_attn_1_0_output)
        if key in {(0, 0)}:
            return 0
        return 7

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_2_0_outputs, num_attn_1_0_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    # num_mlp_2_2 #################################################
    def num_mlp_2_2(num_attn_1_3_output, num_attn_2_3_output):
        key = (num_attn_1_3_output, num_attn_2_3_output)
        return 27

    num_mlp_2_2_outputs = [
        num_mlp_2_2(k0, k1)
        for k0, k1 in zip(num_attn_1_3_outputs, num_attn_2_3_outputs)
    ]
    num_mlp_2_2_output_scores = classifier_weights.loc[
        [("num_mlp_2_2_outputs", str(v)) for v in num_mlp_2_2_outputs]
    ]

    # num_mlp_2_3 #################################################
    def num_mlp_2_3(num_attn_2_1_output, num_attn_1_2_output):
        key = (num_attn_2_1_output, num_attn_1_2_output)
        return 32

    num_mlp_2_3_outputs = [
        num_mlp_2_3(k0, k1)
        for k0, k1 in zip(num_attn_2_1_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_2_3_output_scores = classifier_weights.loc[
        [("num_mlp_2_3_outputs", str(v)) for v in num_mlp_2_3_outputs]
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
                mlp_0_0_output_scores,
                mlp_0_1_output_scores,
                mlp_0_2_output_scores,
                mlp_0_3_output_scores,
                num_mlp_0_0_output_scores,
                num_mlp_0_1_output_scores,
                num_mlp_0_2_output_scores,
                num_mlp_0_3_output_scores,
                attn_1_0_output_scores,
                attn_1_1_output_scores,
                attn_1_2_output_scores,
                attn_1_3_output_scores,
                mlp_1_0_output_scores,
                mlp_1_1_output_scores,
                mlp_1_2_output_scores,
                mlp_1_3_output_scores,
                num_mlp_1_0_output_scores,
                num_mlp_1_1_output_scores,
                num_mlp_1_2_output_scores,
                num_mlp_1_3_output_scores,
                attn_2_0_output_scores,
                attn_2_1_output_scores,
                attn_2_2_output_scores,
                attn_2_3_output_scores,
                mlp_2_0_output_scores,
                mlp_2_1_output_scores,
                mlp_2_2_output_scores,
                mlp_2_3_output_scores,
                num_mlp_2_0_output_scores,
                num_mlp_2_1_output_scores,
                num_mlp_2_2_output_scores,
                num_mlp_2_3_output_scores,
                one_scores,
                num_attn_0_0_output_scores,
                num_attn_0_1_output_scores,
                num_attn_0_2_output_scores,
                num_attn_0_3_output_scores,
                num_attn_1_0_output_scores,
                num_attn_1_1_output_scores,
                num_attn_1_2_output_scores,
                num_attn_1_3_output_scores,
                num_attn_2_0_output_scores,
                num_attn_2_1_output_scores,
                num_attn_2_2_output_scores,
                num_attn_2_3_output_scores,
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
        [
            "<s>",
            "{",
            "{",
            "{",
            "(",
            "{",
            "}",
            ")",
            "}",
            "}",
            "(",
            ")",
            "{",
            "}",
            "}",
            "(",
            ")",
            "{",
            "}",
            "(",
        ]
    )
)
