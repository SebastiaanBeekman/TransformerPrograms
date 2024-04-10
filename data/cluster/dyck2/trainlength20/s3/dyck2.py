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
        "output/length/rasp/dyck2/trainlength20/s3/dyck2_weights.csv",
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
        if q_position in {0, 28}:
            return k_position == 31
        elif q_position in {1, 3}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {33, 4}:
            return k_position == 3
        elif q_position in {5}:
            return k_position == 4
        elif q_position in {34, 6}:
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
        elif q_position in {17, 39, 31}:
            return k_position == 16
        elif q_position in {18}:
            return k_position == 17
        elif q_position in {19}:
            return k_position == 18
        elif q_position in {20}:
            return k_position == 19
        elif q_position in {21}:
            return k_position == 28
        elif q_position in {22}:
            return k_position == 26
        elif q_position in {27, 30, 23}:
            return k_position == 27
        elif q_position in {24}:
            return k_position == 29
        elif q_position in {25, 38}:
            return k_position == 21
        elif q_position in {26, 35}:
            return k_position == 35
        elif q_position in {29}:
            return k_position == 36
        elif q_position in {32}:
            return k_position == 37
        elif q_position in {36}:
            return k_position == 38
        elif q_position in {37}:
            return k_position == 22

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 26, 21}:
            return k_position == 32
        elif q_position in {1, 2}:
            return k_position == 1
        elif q_position in {3, 5}:
            return k_position == 2
        elif q_position in {4, 6}:
            return k_position == 3
        elif q_position in {7}:
            return k_position == 6
        elif q_position in {8, 16, 18}:
            return k_position == 7
        elif q_position in {9, 17}:
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
            return k_position == 5
        elif q_position in {15}:
            return k_position == 14
        elif q_position in {19}:
            return k_position == 4
        elif q_position in {20}:
            return k_position == 35
        elif q_position in {22}:
            return k_position == 25
        elif q_position in {23}:
            return k_position == 39
        elif q_position in {24, 35, 29}:
            return k_position == 34
        elif q_position in {25}:
            return k_position == 22
        elif q_position in {34, 27}:
            return k_position == 37
        elif q_position in {32, 28}:
            return k_position == 38
        elif q_position in {30}:
            return k_position == 28
        elif q_position in {31}:
            return k_position == 27
        elif q_position in {33}:
            return k_position == 31
        elif q_position in {36}:
            return k_position == 33
        elif q_position in {37}:
            return k_position == 30
        elif q_position in {38, 39}:
            return k_position == 36

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0, 30}:
            return k_position == 25
        elif q_position in {1}:
            return k_position == 36
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {4, 37}:
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
        elif q_position in {20, 21}:
            return k_position == 23
        elif q_position in {24, 27, 22}:
            return k_position == 21
        elif q_position in {23}:
            return k_position == 20
        elif q_position in {25}:
            return k_position == 33
        elif q_position in {26}:
            return k_position == 35
        elif q_position in {35, 28, 38}:
            return k_position == 19
        elif q_position in {29}:
            return k_position == 27
        elif q_position in {31}:
            return k_position == 39
        elif q_position in {32, 33}:
            return k_position == 38
        elif q_position in {34}:
            return k_position == 30
        elif q_position in {36}:
            return k_position == 29
        elif q_position in {39}:
            return k_position == 37

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 38, 30}:
            return k_position == 22
        elif q_position in {1}:
            return k_position == 26
        elif q_position in {2, 34}:
            return k_position == 25
        elif q_position in {33, 3}:
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
        elif q_position in {19}:
            return k_position == 18
        elif q_position in {20}:
            return k_position == 36
        elif q_position in {37, 21}:
            return k_position == 35
        elif q_position in {22}:
            return k_position == 32
        elif q_position in {26, 23}:
            return k_position == 34
        elif q_position in {24}:
            return k_position == 28
        elif q_position in {25}:
            return k_position == 38
        elif q_position in {27}:
            return k_position == 33
        elif q_position in {28}:
            return k_position == 19
        elif q_position in {35, 29, 31}:
            return k_position == 21
        elif q_position in {32}:
            return k_position == 24
        elif q_position in {36}:
            return k_position == 29
        elif q_position in {39}:
            return k_position == 20

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(token, position):
        if token in {"(", "{"}:
            return position == 8
        elif token in {")"}:
            return position == 0
        elif token in {"<s>"}:
            return position == 9
        elif token in {"}"}:
            return position == 1

    num_attn_0_0_pattern = select(positions, tokens, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_token, k_token):
        if q_token in {"(", "{"}:
            return k_token == ""
        elif q_token in {"<s>", ")"}:
            return k_token == ")"
        elif q_token in {"}"}:
            return k_token == "}"

    num_attn_0_1_pattern = select(tokens, tokens, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(position, token):
        if position in {0, 17, 35, 33}:
            return token == ")"
        elif position in {
            1,
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
            18,
            19,
            21,
            24,
            25,
            26,
            27,
            29,
            31,
            32,
            34,
            36,
            37,
            38,
            39,
        }:
            return token == ""
        elif position in {3, 6}:
            return token == "{"
        elif position in {20, 22, 23, 28, 30}:
            return token == "}"

    num_attn_0_2_pattern = select(tokens, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(q_token, k_token):
        if q_token in {"(", "{"}:
            return k_token == ""
        elif q_token in {")"}:
            return k_token == "}"
        elif q_token in {"<s>"}:
            return k_token == "{"
        elif q_token in {"}"}:
            return k_token == ")"

    num_attn_0_3_pattern = select(tokens, tokens, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_2_output, attn_0_1_output):
        key = (attn_0_2_output, attn_0_1_output)
        if key in {
            (")", ")"),
            (")", "<s>"),
            (")", "}"),
            ("<s>", ")"),
            ("<s>", "}"),
            ("}", ")"),
            ("}", "<s>"),
            ("}", "}"),
        }:
            return 5
        elif key in {("<s>", "("), ("<s>", "<s>"), ("<s>", "{")}:
            return 1
        elif key in {("}", "{")}:
            return 32
        return 4

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_0_1_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_0_output, token):
        key = (attn_0_0_output, token)
        if key in {
            (")", ")"),
            (")", "<s>"),
            (")", "}"),
            ("<s>", "}"),
            ("{", ")"),
            ("{", "}"),
            ("}", ")"),
            ("}", "<s>"),
            ("}", "{"),
            ("}", "}"),
        }:
            return 20
        elif key in {
            ("(", "("),
            ("(", "<s>"),
            ("(", "{"),
            ("<s>", "("),
            ("<s>", "<s>"),
            ("<s>", "{"),
            ("{", "("),
            ("{", "<s>"),
            ("{", "{"),
        }:
            return 34
        elif key in {("<s>", ")")}:
            return 5
        return 33

    mlp_0_1_outputs = [mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_0_outputs, tokens)]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # mlp_0_2 #####################################################
    def mlp_0_2(token, attn_0_0_output):
        key = (token, attn_0_0_output)
        if key in {
            ("(", "("),
            ("(", "<s>"),
            ("(", "{"),
            ("<s>", "("),
            ("<s>", "<s>"),
            ("<s>", "{"),
            ("{", "("),
            ("{", "<s>"),
            ("{", "{"),
        }:
            return 34
        elif key in {(")", "{"), ("}", "(")}:
            return 35
        elif key in {("(", ")"), ("(", "}"), ("<s>", ")")}:
            return 22
        elif key in {(")", "<s>"), ("}", "<s>")}:
            return 2
        elif key in {("}", "{")}:
            return 8
        elif key in {(")", "(")}:
            return 19
        return 20

    mlp_0_2_outputs = [mlp_0_2(k0, k1) for k0, k1 in zip(tokens, attn_0_0_outputs)]
    mlp_0_2_output_scores = classifier_weights.loc[
        [("mlp_0_2_outputs", str(v)) for v in mlp_0_2_outputs]
    ]

    # mlp_0_3 #####################################################
    def mlp_0_3(attn_0_2_output, token):
        key = (attn_0_2_output, token)
        if key in {
            ("(", "("),
            ("(", "<s>"),
            ("(", "{"),
            ("<s>", "("),
            ("<s>", "<s>"),
            ("<s>", "{"),
            ("{", "("),
            ("{", "<s>"),
            ("{", "{"),
        }:
            return 1
        elif key in {("{", "}")}:
            return 13
        elif key in {
            ("(", "}"),
            (")", ")"),
            (")", "}"),
            ("<s>", ")"),
            ("<s>", "}"),
            ("{", ")"),
            ("}", ")"),
            ("}", "}"),
        }:
            return 5
        elif key in {("}", "<s>")}:
            return 32
        elif key in {(")", "("), (")", "<s>"), ("}", "(")}:
            return 8
        return 29

    mlp_0_3_outputs = [mlp_0_3(k0, k1) for k0, k1 in zip(attn_0_2_outputs, tokens)]
    mlp_0_3_output_scores = classifier_weights.loc[
        [("mlp_0_3_outputs", str(v)) for v in mlp_0_3_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_1_output, num_attn_0_0_output):
        key = (num_attn_0_1_output, num_attn_0_0_output)
        return 28

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_2_output, num_attn_0_0_output):
        key = (num_attn_0_2_output, num_attn_0_0_output)
        return 3

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # num_mlp_0_2 #################################################
    def num_mlp_0_2(num_attn_0_0_output, num_attn_0_3_output):
        key = (num_attn_0_0_output, num_attn_0_3_output)
        return 14

    num_mlp_0_2_outputs = [
        num_mlp_0_2(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_0_2_output_scores = classifier_weights.loc[
        [("num_mlp_0_2_outputs", str(v)) for v in num_mlp_0_2_outputs]
    ]

    # num_mlp_0_3 #################################################
    def num_mlp_0_3(num_attn_0_1_output, num_attn_0_0_output):
        key = (num_attn_0_1_output, num_attn_0_0_output)
        return 30

    num_mlp_0_3_outputs = [
        num_mlp_0_3(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_0_3_output_scores = classifier_weights.loc[
        [("num_mlp_0_3_outputs", str(v)) for v in num_mlp_0_3_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(token, position):
        if token in {"(", "{"}:
            return position == 2
        elif token in {"<s>", ")", "}"}:
            return position == 4

    attn_1_0_pattern = select_closest(positions, tokens, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, mlp_0_1_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_position, k_position):
        if q_position in {
            0,
            32,
            33,
            34,
            36,
            37,
            38,
            10,
            12,
            14,
            16,
            18,
            24,
            25,
            26,
            30,
        }:
            return k_position == 6
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2, 31, 39, 23}:
            return k_position == 2
        elif q_position in {8, 17, 3, 6}:
            return k_position == 5
        elif q_position in {4, 7, 9, 11, 13, 15, 19, 22, 28}:
            return k_position == 3
        elif q_position in {5}:
            return k_position == 4
        elif q_position in {20}:
            return k_position == 27
        elif q_position in {35, 29, 27, 21}:
            return k_position == 34

    attn_1_1_pattern = select_closest(positions, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, mlp_0_3_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(token, mlp_0_3_output):
        if token in {")", "(", "{", "}"}:
            return mlp_0_3_output == 5
        elif token in {"<s>"}:
            return mlp_0_3_output == 4

    attn_1_2_pattern = select_closest(mlp_0_3_outputs, tokens, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, mlp_0_1_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(attn_0_2_output, mlp_0_3_output):
        if attn_0_2_output in {"<s>", "(", "{", ")", "}"}:
            return mlp_0_3_output == 5

    attn_1_3_pattern = select_closest(mlp_0_3_outputs, attn_0_2_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, mlp_0_2_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(num_mlp_0_0_output, token):
        if num_mlp_0_0_output in {
            0,
            4,
            6,
            9,
            10,
            11,
            14,
            17,
            19,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            31,
            32,
            36,
            37,
            38,
        }:
            return token == "("
        elif num_mlp_0_0_output in {
            1,
            2,
            3,
            33,
            5,
            34,
            7,
            8,
            35,
            39,
            12,
            13,
            15,
            16,
            20,
            29,
            30,
        }:
            return token == "{"
        elif num_mlp_0_0_output in {18}:
            return token == ""

    num_attn_1_0_pattern = select(tokens, num_mlp_0_0_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_0_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(attn_0_1_output, mlp_0_2_output):
        if attn_0_1_output in {"<s>", "(", "{", ")", "}"}:
            return mlp_0_2_output == 35

    num_attn_1_1_pattern = select(mlp_0_2_outputs, attn_0_1_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_0_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(q_mlp_0_0_output, k_mlp_0_0_output):
        if q_mlp_0_0_output in {0, 32, 7, 24, 26}:
            return k_mlp_0_0_output == 21
        elif q_mlp_0_0_output in {1}:
            return k_mlp_0_0_output == 20
        elif q_mlp_0_0_output in {
            2,
            6,
            38,
            8,
            9,
            39,
            11,
            12,
            15,
            17,
            20,
            28,
            29,
            30,
            31,
        }:
            return k_mlp_0_0_output == 14
        elif q_mlp_0_0_output in {3, 18, 19, 22, 27}:
            return k_mlp_0_0_output == 29
        elif q_mlp_0_0_output in {4}:
            return k_mlp_0_0_output == 7
        elif q_mlp_0_0_output in {37, 10, 21, 5}:
            return k_mlp_0_0_output == 18
        elif q_mlp_0_0_output in {25, 13, 33}:
            return k_mlp_0_0_output == 22
        elif q_mlp_0_0_output in {34, 14}:
            return k_mlp_0_0_output == 3
        elif q_mlp_0_0_output in {16, 35, 36}:
            return k_mlp_0_0_output == 2
        elif q_mlp_0_0_output in {23}:
            return k_mlp_0_0_output == 16

    num_attn_1_2_pattern = select(mlp_0_0_outputs, mlp_0_0_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_2_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(num_mlp_0_1_output, mlp_0_0_output):
        if num_mlp_0_1_output in {
            0,
            1,
            2,
            3,
            4,
            7,
            8,
            9,
            11,
            12,
            13,
            14,
            16,
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
        }:
            return mlp_0_0_output == 5
        elif num_mlp_0_1_output in {32, 34, 5, 6, 39, 15}:
            return mlp_0_0_output == 2
        elif num_mlp_0_1_output in {10, 18}:
            return mlp_0_0_output == 20
        elif num_mlp_0_1_output in {17, 19}:
            return mlp_0_0_output == 25

    num_attn_1_3_pattern = select(
        mlp_0_0_outputs, num_mlp_0_1_outputs, num_predicate_1_3
    )
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_1_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_0_output):
        key = attn_1_0_output
        if key in {4, 6, 7, 9, 10, 11, 21, 28, 30, 35, 36, 37}:
            return 3
        elif key in {0, 3, 20, 29, 32, 38, 39}:
            return 27
        elif key in {17, 24, 26, 27}:
            return 12
        elif key in {8, 14}:
            return 35
        elif key in {1}:
            return 31
        return 2

    mlp_1_0_outputs = [mlp_1_0(k0) for k0 in attn_1_0_outputs]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_1_output, attn_1_0_output):
        key = (attn_1_1_output, attn_1_0_output)
        if key in {
            (1, 25),
            (1, 30),
            (1, 35),
            (3, 25),
            (4, 24),
            (4, 25),
            (6, 25),
            (10, 25),
            (11, 25),
            (19, 25),
            (21, 25),
            (24, 4),
            (24, 24),
            (24, 25),
            (24, 37),
            (24, 39),
            (25, 1),
            (25, 3),
            (25, 4),
            (25, 8),
            (25, 9),
            (25, 24),
            (25, 25),
            (25, 30),
            (25, 31),
            (25, 35),
            (25, 37),
            (25, 39),
            (30, 4),
            (30, 9),
            (30, 24),
            (30, 25),
            (30, 37),
            (30, 39),
            (31, 25),
            (34, 25),
            (34, 30),
            (35, 25),
            (37, 4),
            (37, 24),
            (37, 25),
            (37, 37),
            (39, 25),
        }:
            return 0
        elif key in {
            (1, 1),
            (1, 3),
            (1, 4),
            (1, 6),
            (1, 7),
            (1, 8),
            (1, 9),
            (1, 10),
            (1, 15),
            (1, 23),
            (1, 24),
            (1, 31),
            (1, 32),
            (1, 34),
            (1, 36),
            (1, 37),
            (1, 39),
            (3, 9),
            (3, 34),
            (4, 1),
            (4, 4),
            (4, 8),
            (4, 9),
            (4, 34),
            (4, 37),
            (4, 39),
            (6, 34),
            (7, 9),
            (11, 9),
            (24, 9),
            (34, 1),
            (34, 9),
            (34, 24),
            (34, 34),
            (34, 37),
            (37, 9),
            (39, 9),
            (39, 34),
        }:
            return 36
        elif key in {(1, 20), (1, 27)}:
            return 17
        elif key in {(34, 4)}:
            return 9
        return 5

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_1_outputs, attn_1_0_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # mlp_1_2 #####################################################
    def mlp_1_2(token, position):
        key = (token, position)
        if key in {
            ("(", 0),
            ("(", 6),
            ("(", 20),
            ("(", 22),
            ("(", 23),
            ("(", 24),
            ("(", 25),
            ("(", 29),
            ("(", 31),
            ("(", 35),
            ("(", 39),
            (")", 1),
            (")", 6),
            (")", 8),
            (")", 14),
            ("<s>", 0),
            ("<s>", 1),
            ("<s>", 3),
            ("<s>", 6),
            ("<s>", 8),
            ("<s>", 14),
            ("<s>", 20),
            ("<s>", 22),
            ("<s>", 23),
            ("<s>", 24),
            ("<s>", 25),
            ("<s>", 26),
            ("<s>", 27),
            ("<s>", 28),
            ("<s>", 29),
            ("<s>", 30),
            ("<s>", 31),
            ("<s>", 32),
            ("<s>", 33),
            ("<s>", 34),
            ("<s>", 35),
            ("<s>", 36),
            ("<s>", 37),
            ("<s>", 38),
            ("<s>", 39),
            ("{", 0),
            ("{", 6),
            ("{", 14),
            ("{", 20),
            ("{", 21),
            ("{", 22),
            ("{", 23),
            ("{", 24),
            ("{", 25),
            ("{", 26),
            ("{", 27),
            ("{", 28),
            ("{", 29),
            ("{", 30),
            ("{", 31),
            ("{", 32),
            ("{", 33),
            ("{", 34),
            ("{", 35),
            ("{", 36),
            ("{", 37),
            ("{", 38),
            ("{", 39),
            ("}", 1),
            ("}", 6),
            ("}", 8),
            ("}", 14),
            ("}", 22),
            ("}", 26),
        }:
            return 37
        elif key in {
            ("(", 7),
            ("(", 9),
            ("(", 10),
            ("(", 12),
            ("(", 26),
            ("(", 27),
            ("(", 33),
            ("(", 36),
            ("(", 38),
            (")", 7),
            (")", 9),
            (")", 10),
            (")", 12),
            ("<s>", 7),
            ("<s>", 9),
            ("<s>", 10),
            ("<s>", 12),
            ("{", 7),
            ("{", 9),
            ("{", 10),
            ("{", 12),
            ("}", 7),
            ("}", 9),
            ("}", 10),
            ("}", 12),
        }:
            return 10
        elif key in {(")", 2), (")", 3), ("<s>", 2), ("}", 2), ("}", 3)}:
            return 2
        elif key in {
            ("(", 8),
            ("(", 11),
            ("(", 13),
            ("(", 14),
            ("(", 15),
            ("(", 16),
            ("(", 17),
            ("(", 18),
            ("(", 19),
            ("{", 8),
            ("{", 17),
        }:
            return 11
        elif key in {
            ("(", 1),
            ("(", 3),
            ("(", 4),
            ("<s>", 4),
            ("{", 1),
            ("{", 3),
            ("{", 4),
        }:
            return 26
        elif key in {
            ("(", 2),
            ("(", 21),
            ("(", 28),
            ("(", 30),
            ("(", 32),
            ("(", 34),
            ("(", 37),
            ("{", 2),
        }:
            return 29
        return 14

    mlp_1_2_outputs = [mlp_1_2(k0, k1) for k0, k1 in zip(tokens, positions)]
    mlp_1_2_output_scores = classifier_weights.loc[
        [("mlp_1_2_outputs", str(v)) for v in mlp_1_2_outputs]
    ]

    # mlp_1_3 #####################################################
    def mlp_1_3(attn_0_1_output, attn_1_1_output):
        key = (attn_0_1_output, attn_1_1_output)
        if key in {
            (")", 26),
            (")", 33),
            ("<s>", 4),
            ("<s>", 7),
            ("<s>", 10),
            ("<s>", 13),
            ("<s>", 15),
            ("<s>", 19),
            ("<s>", 26),
            ("<s>", 27),
            ("<s>", 33),
            ("<s>", 39),
            ("}", 0),
            ("}", 3),
            ("}", 4),
            ("}", 6),
            ("}", 7),
            ("}", 10),
            ("}", 11),
            ("}", 13),
            ("}", 15),
            ("}", 16),
            ("}", 17),
            ("}", 19),
            ("}", 21),
            ("}", 23),
            ("}", 24),
            ("}", 26),
            ("}", 27),
            ("}", 30),
            ("}", 31),
            ("}", 33),
            ("}", 34),
            ("}", 35),
            ("}", 36),
            ("}", 37),
            ("}", 38),
            ("}", 39),
        }:
            return 10
        elif key in {
            ("(", 29),
            (")", 0),
            (")", 12),
            (")", 14),
            (")", 18),
            (")", 20),
            (")", 22),
            (")", 29),
            (")", 38),
            ("<s>", 0),
            ("<s>", 3),
            ("<s>", 5),
            ("<s>", 6),
            ("<s>", 12),
            ("<s>", 14),
            ("<s>", 18),
            ("<s>", 20),
            ("<s>", 21),
            ("<s>", 22),
            ("<s>", 24),
            ("<s>", 28),
            ("<s>", 29),
            ("<s>", 30),
            ("<s>", 31),
            ("<s>", 32),
            ("<s>", 35),
            ("<s>", 36),
            ("<s>", 38),
            ("{", 29),
            ("}", 5),
            ("}", 12),
            ("}", 14),
            ("}", 18),
            ("}", 20),
            ("}", 22),
            ("}", 28),
            ("}", 29),
            ("}", 32),
        }:
            return 20
        elif key in {("(", 25), (")", 25), ("<s>", 25), ("}", 25)}:
            return 24
        return 9

    mlp_1_3_outputs = [
        mlp_1_3(k0, k1) for k0, k1 in zip(attn_0_1_outputs, attn_1_1_outputs)
    ]
    mlp_1_3_output_scores = classifier_weights.loc[
        [("mlp_1_3_outputs", str(v)) for v in mlp_1_3_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_0_output, num_attn_1_3_output):
        key = (num_attn_1_0_output, num_attn_1_3_output)
        return 14

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_1_3_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_1_output, num_attn_1_0_output):
        key = (num_attn_1_1_output, num_attn_1_0_output)
        return 0

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_1_0_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # num_mlp_1_2 #################################################
    def num_mlp_1_2(num_attn_1_3_output, num_attn_0_2_output):
        key = (num_attn_1_3_output, num_attn_0_2_output)
        return 26

    num_mlp_1_2_outputs = [
        num_mlp_1_2(k0, k1)
        for k0, k1 in zip(num_attn_1_3_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_1_2_output_scores = classifier_weights.loc[
        [("num_mlp_1_2_outputs", str(v)) for v in num_mlp_1_2_outputs]
    ]

    # num_mlp_1_3 #################################################
    def num_mlp_1_3(num_attn_1_3_output, num_attn_1_2_output):
        key = (num_attn_1_3_output, num_attn_1_2_output)
        return 0

    num_mlp_1_3_outputs = [
        num_mlp_1_3(k0, k1)
        for k0, k1 in zip(num_attn_1_3_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_1_3_output_scores = classifier_weights.loc[
        [("num_mlp_1_3_outputs", str(v)) for v in num_mlp_1_3_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(token, position):
        if token in {"<s>", ")", "(", "{"}:
            return position == 5
        elif token in {"}"}:
            return position == 6

    attn_2_0_pattern = select_closest(positions, tokens, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_3_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(token, position):
        if token in {"(", "{"}:
            return position == 1
        elif token in {")", "}"}:
            return position == 3
        elif token in {"<s>"}:
            return position == 6

    attn_2_1_pattern = select_closest(positions, tokens, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_2_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(token, position):
        if token in {"<s>", "("}:
            return position == 5
        elif token in {")", "}"}:
            return position == 4
        elif token in {"{"}:
            return position == 6

    attn_2_2_pattern = select_closest(positions, tokens, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_1_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(attn_0_2_output, position):
        if attn_0_2_output in {"<s>", "(", "{"}:
            return position == 5
        elif attn_0_2_output in {")", "}"}:
            return position == 4

    attn_2_3_pattern = select_closest(positions, attn_0_2_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_0_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(mlp_1_1_output, mlp_0_0_output):
        if mlp_1_1_output in {
            0,
            4,
            7,
            8,
            10,
            11,
            12,
            13,
            14,
            16,
            21,
            23,
            24,
            29,
            30,
            31,
            33,
            36,
            37,
            39,
        }:
            return mlp_0_0_output == 2
        elif mlp_1_1_output in {1}:
            return mlp_0_0_output == 27
        elif mlp_1_1_output in {2, 3, 5, 6, 38, 18, 20, 22, 26}:
            return mlp_0_0_output == 5
        elif mlp_1_1_output in {32, 9, 17, 25}:
            return mlp_0_0_output == 36
        elif mlp_1_1_output in {34, 15}:
            return mlp_0_0_output == 20
        elif mlp_1_1_output in {27, 19}:
            return mlp_0_0_output == 35
        elif mlp_1_1_output in {28}:
            return mlp_0_0_output == 15
        elif mlp_1_1_output in {35}:
            return mlp_0_0_output == 37

    num_attn_2_0_pattern = select(mlp_0_0_outputs, mlp_1_1_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_3_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_1_2_output, mlp_0_2_output):
        if attn_1_2_output in {0, 39, 15, 24, 27, 29, 31}:
            return mlp_0_2_output == 25
        elif attn_1_2_output in {1, 34, 35, 37, 6, 14, 17, 19, 21}:
            return mlp_0_2_output == 35
        elif attn_1_2_output in {
            32,
            2,
            3,
            4,
            5,
            36,
            9,
            10,
            12,
            16,
            18,
            22,
            23,
            25,
            26,
            30,
        }:
            return mlp_0_2_output == 2
        elif attn_1_2_output in {33, 38, 7, 8, 11, 13, 20, 28}:
            return mlp_0_2_output == 34

    num_attn_2_1_pattern = select(mlp_0_2_outputs, attn_1_2_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_0_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(mlp_0_2_output, mlp_1_1_output):
        if mlp_0_2_output in {0, 5, 6, 13, 18, 24, 29}:
            return mlp_1_1_output == 34
        elif mlp_0_2_output in {1, 35, 4}:
            return mlp_1_1_output == 29
        elif mlp_0_2_output in {32, 2, 36, 7, 16, 21, 22, 27, 31}:
            return mlp_1_1_output == 1
        elif mlp_0_2_output in {33, 3, 38, 39, 10, 11, 14, 15, 17, 23, 28, 30}:
            return mlp_1_1_output == 9
        elif mlp_0_2_output in {8}:
            return mlp_1_1_output == 32
        elif mlp_0_2_output in {9}:
            return mlp_1_1_output == 38
        elif mlp_0_2_output in {12}:
            return mlp_1_1_output == 33
        elif mlp_0_2_output in {19}:
            return mlp_1_1_output == 7
        elif mlp_0_2_output in {20}:
            return mlp_1_1_output == 36
        elif mlp_0_2_output in {25}:
            return mlp_1_1_output == 22
        elif mlp_0_2_output in {26}:
            return mlp_1_1_output == 27
        elif mlp_0_2_output in {34, 37}:
            return mlp_1_1_output == 18

    num_attn_2_2_pattern = select(mlp_1_1_outputs, mlp_0_2_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_1_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_1_0_output, attn_1_2_output):
        if attn_1_0_output in {0, 2, 36, 5, 6, 39, 12, 16, 17}:
            return attn_1_2_output == 1
        elif attn_1_0_output in {1, 34, 4, 19, 30}:
            return attn_1_2_output == 7
        elif attn_1_0_output in {32, 3, 7, 8, 11, 15, 20, 21, 22, 26, 28, 29}:
            return attn_1_2_output == 9
        elif attn_1_0_output in {9, 37, 33}:
            return attn_1_2_output == 4
        elif attn_1_0_output in {38, 10, 14, 18, 23, 24, 27}:
            return attn_1_2_output == 34
        elif attn_1_0_output in {13}:
            return attn_1_2_output == 39
        elif attn_1_0_output in {25}:
            return attn_1_2_output == 18
        elif attn_1_0_output in {31}:
            return attn_1_2_output == 3
        elif attn_1_0_output in {35}:
            return attn_1_2_output == 22

    num_attn_2_3_pattern = select(attn_1_2_outputs, attn_1_0_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_3_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_2_output, attn_1_2_output):
        key = (attn_2_2_output, attn_1_2_output)
        if key in {
            (0, 1),
            (0, 4),
            (0, 8),
            (0, 21),
            (0, 23),
            (0, 24),
            (0, 37),
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (1, 6),
            (1, 7),
            (1, 8),
            (1, 9),
            (1, 10),
            (1, 11),
            (1, 12),
            (1, 13),
            (1, 14),
            (1, 15),
            (1, 16),
            (1, 17),
            (1, 18),
            (1, 19),
            (1, 20),
            (1, 21),
            (1, 22),
            (1, 23),
            (1, 24),
            (1, 25),
            (1, 26),
            (1, 27),
            (1, 28),
            (1, 29),
            (1, 30),
            (1, 31),
            (1, 32),
            (1, 33),
            (1, 34),
            (1, 35),
            (1, 36),
            (1, 37),
            (1, 38),
            (1, 39),
            (2, 1),
            (2, 4),
            (2, 8),
            (2, 15),
            (2, 16),
            (2, 21),
            (2, 23),
            (2, 24),
            (2, 34),
            (2, 37),
            (2, 39),
            (3, 8),
            (3, 21),
            (3, 23),
            (3, 24),
            (3, 37),
            (4, 0),
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 4),
            (4, 5),
            (4, 6),
            (4, 7),
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
            (4, 35),
            (4, 36),
            (4, 37),
            (4, 38),
            (4, 39),
            (5, 1),
            (5, 4),
            (5, 8),
            (5, 15),
            (5, 16),
            (5, 21),
            (5, 23),
            (5, 24),
            (5, 34),
            (5, 37),
            (5, 39),
            (6, 1),
            (6, 4),
            (6, 6),
            (6, 8),
            (6, 10),
            (6, 11),
            (6, 12),
            (6, 13),
            (6, 15),
            (6, 16),
            (6, 17),
            (6, 19),
            (6, 21),
            (6, 23),
            (6, 24),
            (6, 25),
            (6, 26),
            (6, 27),
            (6, 28),
            (6, 30),
            (6, 31),
            (6, 33),
            (6, 36),
            (6, 37),
            (6, 39),
            (7, 1),
            (7, 4),
            (7, 6),
            (7, 8),
            (7, 15),
            (7, 16),
            (7, 19),
            (7, 21),
            (7, 23),
            (7, 24),
            (7, 27),
            (7, 28),
            (7, 36),
            (7, 37),
            (7, 39),
            (9, 8),
            (10, 1),
            (10, 4),
            (10, 6),
            (10, 8),
            (10, 10),
            (10, 11),
            (10, 15),
            (10, 16),
            (10, 19),
            (10, 21),
            (10, 23),
            (10, 24),
            (10, 25),
            (10, 26),
            (10, 27),
            (10, 28),
            (10, 31),
            (10, 36),
            (10, 37),
            (10, 39),
            (11, 1),
            (11, 4),
            (11, 8),
            (11, 21),
            (11, 23),
            (11, 24),
            (11, 37),
            (12, 1),
            (12, 4),
            (12, 8),
            (12, 21),
            (12, 23),
            (12, 24),
            (12, 37),
            (13, 1),
            (13, 2),
            (13, 3),
            (13, 4),
            (13, 5),
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
            (13, 16),
            (13, 17),
            (13, 18),
            (13, 19),
            (13, 20),
            (13, 21),
            (13, 23),
            (13, 24),
            (13, 25),
            (13, 26),
            (13, 27),
            (13, 28),
            (13, 29),
            (13, 30),
            (13, 31),
            (13, 32),
            (13, 33),
            (13, 34),
            (13, 35),
            (13, 36),
            (13, 37),
            (13, 38),
            (13, 39),
            (14, 1),
            (14, 4),
            (14, 8),
            (14, 15),
            (14, 21),
            (14, 23),
            (14, 24),
            (14, 37),
            (15, 1),
            (15, 4),
            (15, 6),
            (15, 8),
            (15, 9),
            (15, 10),
            (15, 11),
            (15, 12),
            (15, 13),
            (15, 15),
            (15, 16),
            (15, 17),
            (15, 19),
            (15, 21),
            (15, 23),
            (15, 24),
            (15, 25),
            (15, 26),
            (15, 27),
            (15, 28),
            (15, 29),
            (15, 30),
            (15, 31),
            (15, 33),
            (15, 36),
            (15, 37),
            (15, 38),
            (15, 39),
            (16, 1),
            (16, 4),
            (16, 8),
            (16, 15),
            (16, 21),
            (16, 23),
            (16, 24),
            (16, 37),
            (17, 1),
            (17, 4),
            (17, 6),
            (17, 8),
            (17, 10),
            (17, 11),
            (17, 13),
            (17, 15),
            (17, 16),
            (17, 17),
            (17, 19),
            (17, 21),
            (17, 23),
            (17, 24),
            (17, 25),
            (17, 26),
            (17, 27),
            (17, 28),
            (17, 30),
            (17, 31),
            (17, 33),
            (17, 36),
            (17, 37),
            (17, 39),
            (18, 1),
            (18, 4),
            (18, 8),
            (19, 1),
            (19, 4),
            (19, 6),
            (19, 8),
            (19, 9),
            (19, 10),
            (19, 11),
            (19, 12),
            (19, 13),
            (19, 15),
            (19, 16),
            (19, 17),
            (19, 19),
            (19, 21),
            (19, 23),
            (19, 24),
            (19, 25),
            (19, 26),
            (19, 27),
            (19, 28),
            (19, 30),
            (19, 31),
            (19, 33),
            (19, 36),
            (19, 37),
            (19, 39),
            (20, 1),
            (20, 4),
            (20, 34),
            (21, 1),
            (21, 4),
            (21, 6),
            (21, 8),
            (21, 15),
            (21, 16),
            (21, 19),
            (21, 21),
            (21, 23),
            (21, 24),
            (21, 27),
            (21, 28),
            (21, 36),
            (21, 37),
            (21, 39),
            (22, 1),
            (22, 4),
            (23, 1),
            (23, 4),
            (23, 8),
            (23, 15),
            (23, 21),
            (23, 23),
            (23, 24),
            (23, 37),
            (24, 1),
            (24, 4),
            (24, 8),
            (24, 15),
            (24, 21),
            (24, 23),
            (24, 24),
            (24, 37),
            (25, 1),
            (25, 4),
            (25, 8),
            (25, 15),
            (25, 16),
            (25, 21),
            (25, 23),
            (25, 24),
            (25, 37),
            (27, 6),
            (27, 8),
            (27, 10),
            (27, 11),
            (27, 13),
            (27, 15),
            (27, 16),
            (27, 17),
            (27, 19),
            (27, 21),
            (27, 23),
            (27, 24),
            (27, 25),
            (27, 26),
            (27, 27),
            (27, 28),
            (27, 30),
            (27, 31),
            (27, 33),
            (27, 36),
            (27, 37),
            (27, 39),
            (28, 1),
            (28, 2),
            (28, 3),
            (28, 4),
            (28, 5),
            (28, 6),
            (28, 7),
            (28, 8),
            (28, 9),
            (28, 10),
            (28, 11),
            (28, 12),
            (28, 13),
            (28, 14),
            (28, 15),
            (28, 16),
            (28, 17),
            (28, 18),
            (28, 19),
            (28, 20),
            (28, 21),
            (28, 23),
            (28, 24),
            (28, 25),
            (28, 26),
            (28, 27),
            (28, 28),
            (28, 29),
            (28, 30),
            (28, 31),
            (28, 32),
            (28, 33),
            (28, 35),
            (28, 36),
            (28, 37),
            (28, 38),
            (28, 39),
            (30, 1),
            (30, 4),
            (30, 6),
            (30, 8),
            (30, 15),
            (30, 16),
            (30, 19),
            (30, 21),
            (30, 23),
            (30, 24),
            (30, 26),
            (30, 27),
            (30, 28),
            (30, 36),
            (30, 37),
            (30, 39),
            (31, 1),
            (31, 4),
            (31, 8),
            (31, 15),
            (31, 21),
            (31, 23),
            (31, 24),
            (31, 37),
            (32, 1),
            (32, 4),
            (32, 6),
            (32, 8),
            (32, 15),
            (32, 16),
            (32, 19),
            (32, 21),
            (32, 23),
            (32, 24),
            (32, 27),
            (32, 28),
            (32, 36),
            (32, 37),
            (32, 39),
            (33, 1),
            (33, 4),
            (33, 34),
            (34, 0),
            (34, 2),
            (34, 3),
            (34, 5),
            (34, 6),
            (34, 7),
            (34, 8),
            (34, 10),
            (34, 11),
            (34, 12),
            (34, 13),
            (34, 14),
            (34, 15),
            (34, 16),
            (34, 17),
            (34, 18),
            (34, 19),
            (34, 20),
            (34, 21),
            (34, 22),
            (34, 23),
            (34, 24),
            (34, 25),
            (34, 26),
            (34, 27),
            (34, 28),
            (34, 29),
            (34, 30),
            (34, 31),
            (34, 32),
            (34, 33),
            (34, 35),
            (34, 36),
            (34, 37),
            (34, 39),
            (35, 1),
            (35, 4),
            (35, 6),
            (35, 8),
            (35, 15),
            (35, 16),
            (35, 19),
            (35, 21),
            (35, 23),
            (35, 24),
            (35, 27),
            (35, 28),
            (35, 36),
            (35, 37),
            (35, 39),
            (36, 1),
            (36, 4),
            (36, 8),
            (36, 15),
            (36, 21),
            (36, 23),
            (36, 24),
            (36, 37),
            (37, 1),
            (37, 4),
            (37, 6),
            (37, 8),
            (37, 10),
            (37, 11),
            (37, 12),
            (37, 13),
            (37, 15),
            (37, 16),
            (37, 17),
            (37, 19),
            (37, 21),
            (37, 23),
            (37, 24),
            (37, 25),
            (37, 26),
            (37, 27),
            (37, 28),
            (37, 30),
            (37, 31),
            (37, 33),
            (37, 36),
            (37, 37),
            (37, 39),
            (38, 1),
            (38, 4),
            (38, 6),
            (38, 8),
            (38, 15),
            (38, 16),
            (38, 19),
            (38, 21),
            (38, 23),
            (38, 24),
            (38, 26),
            (38, 27),
            (38, 28),
            (38, 36),
            (38, 37),
            (38, 39),
            (39, 1),
            (39, 4),
            (39, 8),
            (39, 15),
            (39, 21),
            (39, 23),
            (39, 24),
            (39, 37),
        }:
            return 20
        elif key in {
            (0, 34),
            (3, 1),
            (3, 4),
            (3, 9),
            (3, 34),
            (4, 34),
            (6, 34),
            (7, 34),
            (8, 1),
            (8, 4),
            (8, 34),
            (9, 1),
            (9, 4),
            (9, 9),
            (9, 21),
            (9, 34),
            (9, 37),
            (10, 34),
            (11, 34),
            (12, 34),
            (14, 34),
            (15, 34),
            (16, 34),
            (17, 9),
            (17, 34),
            (18, 34),
            (19, 34),
            (21, 34),
            (22, 34),
            (23, 34),
            (24, 34),
            (25, 34),
            (26, 1),
            (26, 4),
            (26, 9),
            (26, 15),
            (26, 21),
            (26, 23),
            (26, 24),
            (26, 34),
            (26, 37),
            (26, 39),
            (27, 1),
            (27, 4),
            (27, 9),
            (27, 34),
            (28, 34),
            (29, 1),
            (29, 4),
            (29, 34),
            (30, 34),
            (31, 34),
            (32, 34),
            (34, 4),
            (34, 9),
            (34, 34),
            (35, 34),
            (36, 34),
            (37, 9),
            (37, 34),
            (38, 34),
            (39, 34),
        }:
            return 28
        elif key in {(6, 9), (26, 8), (34, 1), (34, 38)}:
            return 15
        return 30

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_2_outputs, attn_1_2_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_2_1_output, attn_2_0_output):
        key = (attn_2_1_output, attn_2_0_output)
        if key in {
            (0, 20),
            (1, 0),
            (1, 1),
            (1, 3),
            (1, 4),
            (1, 5),
            (1, 6),
            (1, 7),
            (1, 8),
            (1, 9),
            (1, 13),
            (1, 14),
            (1, 19),
            (1, 20),
            (1, 23),
            (1, 29),
            (1, 30),
            (1, 32),
            (1, 35),
            (1, 36),
            (1, 39),
            (2, 20),
            (2, 29),
            (3, 1),
            (3, 4),
            (3, 5),
            (3, 7),
            (3, 8),
            (3, 9),
            (3, 13),
            (3, 19),
            (3, 20),
            (3, 23),
            (3, 29),
            (3, 30),
            (3, 32),
            (3, 35),
            (3, 36),
            (4, 0),
            (4, 1),
            (4, 3),
            (4, 4),
            (4, 5),
            (4, 6),
            (4, 7),
            (4, 8),
            (4, 9),
            (4, 10),
            (4, 11),
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
            (4, 26),
            (4, 28),
            (4, 29),
            (4, 30),
            (4, 31),
            (4, 32),
            (4, 33),
            (4, 35),
            (4, 36),
            (4, 37),
            (4, 39),
            (5, 1),
            (5, 9),
            (5, 20),
            (5, 29),
            (6, 13),
            (6, 20),
            (6, 29),
            (6, 32),
            (7, 8),
            (7, 9),
            (7, 13),
            (7, 20),
            (7, 29),
            (7, 32),
            (8, 20),
            (9, 0),
            (9, 1),
            (9, 3),
            (9, 4),
            (9, 5),
            (9, 6),
            (9, 7),
            (9, 8),
            (9, 9),
            (9, 11),
            (9, 13),
            (9, 14),
            (9, 15),
            (9, 16),
            (9, 17),
            (9, 19),
            (9, 20),
            (9, 21),
            (9, 22),
            (9, 23),
            (9, 24),
            (9, 26),
            (9, 28),
            (9, 29),
            (9, 30),
            (9, 31),
            (9, 32),
            (9, 33),
            (9, 35),
            (9, 36),
            (9, 37),
            (9, 39),
            (10, 20),
            (11, 13),
            (11, 20),
            (11, 29),
            (11, 32),
            (12, 20),
            (12, 29),
            (13, 1),
            (13, 4),
            (13, 5),
            (13, 7),
            (13, 8),
            (13, 9),
            (13, 13),
            (13, 20),
            (13, 29),
            (13, 30),
            (13, 32),
            (14, 4),
            (14, 5),
            (14, 7),
            (14, 8),
            (14, 9),
            (14, 13),
            (14, 20),
            (14, 29),
            (14, 32),
            (15, 20),
            (15, 29),
            (16, 1),
            (16, 4),
            (16, 5),
            (16, 7),
            (16, 8),
            (16, 9),
            (16, 13),
            (16, 19),
            (16, 20),
            (16, 23),
            (16, 29),
            (16, 30),
            (16, 32),
            (16, 36),
            (17, 20),
            (17, 29),
            (18, 20),
            (19, 0),
            (19, 1),
            (19, 3),
            (19, 4),
            (19, 5),
            (19, 6),
            (19, 7),
            (19, 8),
            (19, 9),
            (19, 13),
            (19, 14),
            (19, 15),
            (19, 16),
            (19, 17),
            (19, 19),
            (19, 20),
            (19, 23),
            (19, 24),
            (19, 28),
            (19, 29),
            (19, 30),
            (19, 31),
            (19, 32),
            (19, 33),
            (19, 35),
            (19, 36),
            (19, 37),
            (19, 39),
            (20, 20),
            (21, 20),
            (22, 20),
            (23, 20),
            (24, 20),
            (25, 8),
            (25, 9),
            (25, 13),
            (25, 20),
            (25, 29),
            (25, 32),
            (26, 20),
            (27, 0),
            (27, 1),
            (27, 3),
            (27, 4),
            (27, 5),
            (27, 6),
            (27, 7),
            (27, 8),
            (27, 9),
            (27, 13),
            (27, 14),
            (27, 16),
            (27, 17),
            (27, 19),
            (27, 20),
            (27, 23),
            (27, 24),
            (27, 29),
            (27, 30),
            (27, 32),
            (27, 33),
            (27, 35),
            (27, 36),
            (27, 39),
            (28, 20),
            (29, 1),
            (29, 4),
            (29, 7),
            (29, 8),
            (29, 9),
            (29, 13),
            (29, 20),
            (29, 29),
            (29, 30),
            (29, 32),
            (30, 1),
            (30, 4),
            (30, 5),
            (30, 7),
            (30, 8),
            (30, 9),
            (30, 13),
            (30, 20),
            (30, 29),
            (30, 30),
            (30, 32),
            (31, 20),
            (32, 13),
            (32, 20),
            (32, 29),
            (32, 32),
            (33, 13),
            (33, 20),
            (33, 29),
            (34, 0),
            (34, 1),
            (34, 2),
            (34, 3),
            (34, 4),
            (34, 5),
            (34, 6),
            (34, 7),
            (34, 8),
            (34, 9),
            (34, 10),
            (34, 11),
            (34, 12),
            (34, 13),
            (34, 14),
            (34, 15),
            (34, 16),
            (34, 17),
            (34, 18),
            (34, 19),
            (34, 20),
            (34, 21),
            (34, 22),
            (34, 23),
            (34, 24),
            (34, 25),
            (34, 26),
            (34, 27),
            (34, 28),
            (34, 29),
            (34, 30),
            (34, 31),
            (34, 32),
            (34, 33),
            (34, 34),
            (34, 35),
            (34, 36),
            (34, 37),
            (34, 38),
            (34, 39),
            (35, 20),
            (35, 29),
            (36, 20),
            (37, 5),
            (37, 13),
            (37, 20),
            (37, 29),
            (37, 32),
            (38, 20),
            (38, 29),
            (39, 20),
        }:
            return 6
        return 11

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_2_1_outputs, attn_2_0_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # mlp_2_2 #####################################################
    def mlp_2_2(position, attn_0_1_output):
        key = (position, attn_0_1_output)
        if key in {
            (0, "{"),
            (1, "{"),
            (2, "{"),
            (3, "{"),
            (4, "("),
            (4, "<s>"),
            (4, "{"),
            (4, "}"),
            (5, "("),
            (5, "<s>"),
            (5, "{"),
            (6, "{"),
            (7, "{"),
            (8, "("),
            (8, ")"),
            (8, "<s>"),
            (8, "{"),
            (8, "}"),
            (9, "{"),
            (10, "{"),
            (11, "{"),
            (12, "{"),
            (13, "("),
            (13, "<s>"),
            (13, "{"),
            (13, "}"),
            (14, "("),
            (14, "<s>"),
            (14, "{"),
            (14, "}"),
            (15, "{"),
            (16, "("),
            (16, ")"),
            (16, "<s>"),
            (16, "{"),
            (16, "}"),
            (17, "{"),
            (18, "{"),
            (19, "("),
            (19, "<s>"),
            (19, "{"),
            (19, "}"),
            (20, "{"),
            (21, "{"),
            (22, "{"),
            (23, "{"),
            (24, "{"),
            (25, "{"),
            (26, "{"),
            (27, "{"),
            (28, "{"),
            (29, "{"),
            (30, "{"),
            (31, "{"),
            (32, "{"),
            (33, "{"),
            (34, ")"),
            (34, "{"),
            (34, "}"),
            (35, "{"),
            (36, "{"),
            (37, "{"),
            (38, "{"),
            (39, "{"),
        }:
            return 28
        elif key in {
            (0, ")"),
            (0, "}"),
            (1, ")"),
            (1, "}"),
            (2, "("),
            (2, ")"),
            (2, "<s>"),
            (2, "}"),
            (3, ")"),
            (3, "}"),
            (4, ")"),
            (5, ")"),
            (5, "}"),
            (6, ")"),
            (6, "<s>"),
            (6, "}"),
            (19, ")"),
            (20, ")"),
            (20, "}"),
            (21, ")"),
            (21, "}"),
            (22, ")"),
            (22, "}"),
            (23, ")"),
            (23, "}"),
            (24, ")"),
            (24, "}"),
            (25, ")"),
            (25, "}"),
            (26, ")"),
            (26, "}"),
            (27, ")"),
            (27, "}"),
            (28, ")"),
            (28, "}"),
            (29, ")"),
            (29, "}"),
            (30, ")"),
            (30, "}"),
            (31, ")"),
            (31, "}"),
            (32, ")"),
            (32, "}"),
            (33, ")"),
            (33, "}"),
            (35, ")"),
            (35, "}"),
            (36, ")"),
            (36, "}"),
            (37, ")"),
            (37, "}"),
            (38, ")"),
            (38, "}"),
            (39, ")"),
            (39, "}"),
        }:
            return 9
        return 30

    mlp_2_2_outputs = [mlp_2_2(k0, k1) for k0, k1 in zip(positions, attn_0_1_outputs)]
    mlp_2_2_output_scores = classifier_weights.loc[
        [("mlp_2_2_outputs", str(v)) for v in mlp_2_2_outputs]
    ]

    # mlp_2_3 #####################################################
    def mlp_2_3(attn_2_0_output, attn_2_3_output):
        key = (attn_2_0_output, attn_2_3_output)
        if key in {
            (0, 34),
            (1, 0),
            (1, 1),
            (1, 6),
            (1, 9),
            (1, 10),
            (1, 15),
            (1, 18),
            (1, 21),
            (1, 24),
            (1, 26),
            (1, 30),
            (1, 31),
            (1, 32),
            (1, 33),
            (1, 34),
            (1, 35),
            (1, 36),
            (1, 37),
            (1, 39),
            (2, 34),
            (3, 1),
            (3, 34),
            (3, 35),
            (4, 1),
            (4, 9),
            (4, 31),
            (4, 34),
            (4, 35),
            (4, 37),
            (5, 1),
            (5, 34),
            (5, 35),
            (6, 1),
            (6, 31),
            (6, 34),
            (6, 35),
            (7, 1),
            (7, 31),
            (7, 34),
            (7, 35),
            (9, 0),
            (9, 1),
            (9, 3),
            (9, 4),
            (9, 6),
            (9, 7),
            (9, 9),
            (9, 10),
            (9, 12),
            (9, 15),
            (9, 16),
            (9, 24),
            (9, 26),
            (9, 30),
            (9, 31),
            (9, 32),
            (9, 34),
            (9, 35),
            (9, 36),
            (9, 37),
            (9, 39),
            (10, 1),
            (10, 31),
            (10, 34),
            (10, 35),
            (11, 1),
            (11, 34),
            (11, 35),
            (12, 1),
            (12, 31),
            (12, 34),
            (12, 35),
            (13, 1),
            (13, 3),
            (13, 4),
            (13, 6),
            (13, 7),
            (13, 9),
            (13, 10),
            (13, 15),
            (13, 16),
            (13, 21),
            (13, 24),
            (13, 26),
            (13, 30),
            (13, 31),
            (13, 34),
            (13, 35),
            (13, 36),
            (13, 37),
            (13, 39),
            (14, 34),
            (15, 1),
            (15, 3),
            (15, 6),
            (15, 9),
            (15, 10),
            (15, 15),
            (15, 18),
            (15, 21),
            (15, 24),
            (15, 26),
            (15, 29),
            (15, 30),
            (15, 31),
            (15, 32),
            (15, 33),
            (15, 34),
            (15, 35),
            (15, 36),
            (15, 37),
            (15, 39),
            (16, 1),
            (16, 34),
            (17, 1),
            (17, 6),
            (17, 9),
            (17, 15),
            (17, 26),
            (17, 30),
            (17, 31),
            (17, 34),
            (17, 35),
            (17, 36),
            (17, 37),
            (18, 1),
            (18, 6),
            (18, 9),
            (18, 15),
            (18, 34),
            (18, 35),
            (18, 37),
            (21, 1),
            (21, 6),
            (21, 9),
            (21, 15),
            (21, 31),
            (21, 34),
            (21, 35),
            (22, 34),
            (23, 1),
            (23, 6),
            (23, 9),
            (23, 10),
            (23, 15),
            (23, 21),
            (23, 24),
            (23, 26),
            (23, 30),
            (23, 31),
            (23, 34),
            (23, 35),
            (23, 36),
            (23, 37),
            (23, 39),
            (24, 1),
            (24, 2),
            (24, 3),
            (24, 4),
            (24, 6),
            (24, 7),
            (24, 9),
            (24, 10),
            (24, 12),
            (24, 15),
            (24, 16),
            (24, 18),
            (24, 21),
            (24, 23),
            (24, 24),
            (24, 25),
            (24, 26),
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
            (25, 1),
            (25, 9),
            (25, 34),
            (25, 35),
            (25, 37),
            (26, 1),
            (26, 6),
            (26, 9),
            (26, 10),
            (26, 15),
            (26, 24),
            (26, 30),
            (26, 31),
            (26, 34),
            (26, 35),
            (26, 36),
            (26, 37),
            (26, 39),
            (27, 1),
            (27, 34),
            (28, 0),
            (28, 1),
            (28, 2),
            (28, 3),
            (28, 4),
            (28, 6),
            (28, 7),
            (28, 9),
            (28, 10),
            (28, 12),
            (28, 15),
            (28, 16),
            (28, 18),
            (28, 21),
            (28, 23),
            (28, 24),
            (28, 25),
            (28, 26),
            (28, 29),
            (28, 30),
            (28, 31),
            (28, 32),
            (28, 33),
            (28, 34),
            (28, 35),
            (28, 36),
            (28, 37),
            (28, 39),
            (29, 1),
            (29, 34),
            (30, 1),
            (30, 9),
            (30, 15),
            (30, 31),
            (30, 34),
            (30, 35),
            (31, 34),
            (32, 1),
            (32, 31),
            (32, 34),
            (32, 35),
            (34, 0),
            (34, 1),
            (34, 2),
            (34, 3),
            (34, 4),
            (34, 6),
            (34, 7),
            (34, 9),
            (34, 10),
            (34, 12),
            (34, 15),
            (34, 16),
            (34, 18),
            (34, 21),
            (34, 23),
            (34, 24),
            (34, 26),
            (34, 29),
            (34, 30),
            (34, 31),
            (34, 32),
            (34, 33),
            (34, 34),
            (34, 35),
            (34, 36),
            (34, 37),
            (34, 39),
            (35, 1),
            (35, 4),
            (35, 6),
            (35, 9),
            (35, 10),
            (35, 15),
            (35, 31),
            (35, 34),
            (35, 35),
            (35, 37),
            (36, 1),
            (36, 34),
            (36, 35),
            (37, 1),
            (37, 2),
            (37, 3),
            (37, 4),
            (37, 6),
            (37, 7),
            (37, 9),
            (37, 10),
            (37, 12),
            (37, 15),
            (37, 16),
            (37, 18),
            (37, 21),
            (37, 23),
            (37, 24),
            (37, 26),
            (37, 30),
            (37, 31),
            (37, 32),
            (37, 33),
            (37, 34),
            (37, 35),
            (37, 36),
            (37, 37),
            (37, 39),
            (38, 1),
            (38, 6),
            (38, 9),
            (38, 10),
            (38, 15),
            (38, 34),
            (38, 35),
            (39, 1),
            (39, 6),
            (39, 9),
            (39, 15),
            (39, 31),
            (39, 34),
            (39, 35),
        }:
            return 23
        elif key in {
            (0, 1),
            (0, 9),
            (0, 15),
            (1, 11),
            (1, 20),
            (1, 22),
            (2, 1),
            (2, 6),
            (2, 9),
            (2, 15),
            (2, 25),
            (3, 9),
            (3, 11),
            (3, 18),
            (3, 21),
            (3, 26),
            (3, 30),
            (3, 31),
            (3, 36),
            (3, 37),
            (3, 39),
            (4, 11),
            (4, 18),
            (4, 20),
            (4, 21),
            (4, 22),
            (4, 24),
            (4, 26),
            (4, 33),
            (5, 3),
            (5, 6),
            (5, 7),
            (5, 9),
            (5, 10),
            (5, 11),
            (5, 15),
            (5, 25),
            (5, 26),
            (5, 30),
            (5, 31),
            (5, 36),
            (5, 37),
            (5, 38),
            (6, 9),
            (6, 11),
            (6, 18),
            (6, 21),
            (6, 24),
            (6, 26),
            (6, 29),
            (6, 30),
            (6, 33),
            (6, 36),
            (6, 37),
            (7, 11),
            (7, 18),
            (7, 21),
            (7, 24),
            (7, 26),
            (7, 30),
            (7, 33),
            (7, 36),
            (7, 37),
            (9, 2),
            (9, 11),
            (9, 17),
            (9, 18),
            (9, 20),
            (9, 21),
            (9, 22),
            (9, 23),
            (9, 25),
            (9, 29),
            (9, 33),
            (9, 38),
            (10, 9),
            (10, 11),
            (10, 15),
            (10, 18),
            (10, 21),
            (10, 24),
            (10, 26),
            (10, 30),
            (10, 36),
            (10, 37),
            (10, 39),
            (11, 6),
            (11, 9),
            (11, 11),
            (11, 15),
            (11, 18),
            (11, 21),
            (11, 22),
            (11, 24),
            (11, 26),
            (11, 30),
            (11, 31),
            (11, 32),
            (11, 33),
            (11, 36),
            (11, 37),
            (11, 39),
            (12, 11),
            (12, 18),
            (12, 21),
            (12, 24),
            (12, 26),
            (12, 37),
            (13, 2),
            (13, 11),
            (13, 17),
            (13, 18),
            (13, 25),
            (13, 38),
            (14, 1),
            (14, 9),
            (14, 15),
            (15, 11),
            (16, 11),
            (16, 18),
            (16, 21),
            (16, 24),
            (16, 26),
            (16, 37),
            (17, 11),
            (17, 21),
            (20, 1),
            (20, 4),
            (20, 9),
            (20, 15),
            (20, 34),
            (21, 2),
            (21, 3),
            (21, 10),
            (21, 11),
            (21, 18),
            (21, 21),
            (21, 23),
            (21, 24),
            (21, 25),
            (21, 26),
            (21, 29),
            (21, 30),
            (21, 32),
            (21, 33),
            (21, 36),
            (21, 37),
            (21, 38),
            (21, 39),
            (23, 11),
            (24, 11),
            (25, 11),
            (25, 18),
            (25, 21),
            (25, 24),
            (25, 26),
            (25, 30),
            (26, 2),
            (26, 11),
            (26, 18),
            (26, 21),
            (26, 23),
            (26, 25),
            (26, 26),
            (26, 29),
            (26, 33),
            (26, 38),
            (27, 11),
            (27, 18),
            (27, 21),
            (27, 26),
            (27, 33),
            (27, 37),
            (28, 11),
            (30, 2),
            (30, 3),
            (30, 6),
            (30, 10),
            (30, 11),
            (30, 18),
            (30, 21),
            (30, 24),
            (30, 25),
            (30, 26),
            (30, 30),
            (30, 36),
            (30, 37),
            (30, 38),
            (30, 39),
            (31, 1),
            (31, 9),
            (31, 11),
            (31, 18),
            (31, 21),
            (31, 24),
            (31, 26),
            (31, 30),
            (31, 36),
            (31, 37),
            (32, 2),
            (32, 9),
            (32, 11),
            (32, 15),
            (32, 18),
            (32, 21),
            (32, 24),
            (32, 26),
            (32, 30),
            (32, 36),
            (32, 37),
            (32, 39),
            (34, 11),
            (34, 20),
            (34, 22),
            (36, 11),
            (36, 18),
            (36, 21),
            (36, 24),
            (36, 26),
            (36, 30),
            (36, 33),
            (36, 37),
            (37, 11),
            (37, 25),
            (38, 2),
            (38, 3),
            (38, 11),
            (38, 18),
            (38, 21),
            (38, 23),
            (38, 24),
            (38, 25),
            (38, 26),
            (38, 29),
            (38, 30),
            (38, 32),
            (38, 33),
            (38, 36),
            (38, 38),
            (38, 39),
            (39, 11),
            (39, 24),
            (39, 25),
            (39, 30),
            (39, 37),
        }:
            return 26
        elif key in {
            (0, 0),
            (0, 2),
            (0, 3),
            (0, 6),
            (0, 10),
            (0, 11),
            (0, 18),
            (0, 20),
            (0, 21),
            (0, 22),
            (0, 23),
            (0, 24),
            (0, 25),
            (0, 26),
            (0, 29),
            (0, 30),
            (0, 31),
            (0, 32),
            (0, 35),
            (0, 36),
            (0, 37),
            (0, 38),
            (0, 39),
            (2, 0),
            (2, 2),
            (2, 3),
            (2, 10),
            (2, 11),
            (2, 16),
            (2, 18),
            (2, 20),
            (2, 21),
            (2, 22),
            (2, 23),
            (2, 24),
            (2, 26),
            (2, 29),
            (2, 30),
            (2, 31),
            (2, 32),
            (2, 35),
            (2, 36),
            (2, 37),
            (2, 38),
            (2, 39),
            (3, 20),
            (3, 22),
            (5, 0),
            (5, 18),
            (5, 20),
            (5, 21),
            (5, 22),
            (5, 29),
            (5, 32),
            (5, 39),
            (6, 20),
            (6, 22),
            (7, 20),
            (7, 22),
            (10, 20),
            (10, 22),
            (11, 20),
            (13, 20),
            (13, 22),
            (13, 29),
            (14, 0),
            (14, 2),
            (14, 11),
            (14, 18),
            (14, 20),
            (14, 21),
            (14, 22),
            (14, 23),
            (14, 24),
            (14, 26),
            (14, 29),
            (14, 30),
            (14, 31),
            (14, 32),
            (14, 36),
            (14, 37),
            (14, 38),
            (14, 39),
            (15, 20),
            (16, 20),
            (17, 20),
            (17, 22),
            (18, 0),
            (18, 2),
            (18, 11),
            (18, 18),
            (18, 20),
            (18, 21),
            (18, 22),
            (18, 23),
            (18, 26),
            (18, 29),
            (18, 30),
            (18, 31),
            (18, 32),
            (18, 36),
            (18, 38),
            (18, 39),
            (19, 21),
            (20, 0),
            (20, 2),
            (20, 3),
            (20, 6),
            (20, 7),
            (20, 8),
            (20, 10),
            (20, 11),
            (20, 12),
            (20, 13),
            (20, 14),
            (20, 16),
            (20, 17),
            (20, 18),
            (20, 20),
            (20, 21),
            (20, 22),
            (20, 23),
            (20, 24),
            (20, 25),
            (20, 26),
            (20, 27),
            (20, 29),
            (20, 30),
            (20, 31),
            (20, 32),
            (20, 35),
            (20, 36),
            (20, 37),
            (20, 38),
            (20, 39),
            (21, 20),
            (21, 22),
            (22, 20),
            (22, 21),
            (22, 22),
            (22, 26),
            (22, 31),
            (23, 20),
            (23, 22),
            (24, 20),
            (24, 22),
            (25, 20),
            (25, 22),
            (26, 20),
            (26, 22),
            (27, 20),
            (28, 20),
            (29, 0),
            (29, 2),
            (29, 3),
            (29, 18),
            (29, 20),
            (29, 21),
            (29, 22),
            (29, 23),
            (29, 26),
            (29, 29),
            (29, 31),
            (29, 32),
            (29, 36),
            (29, 37),
            (29, 38),
            (29, 39),
            (30, 20),
            (30, 22),
            (32, 20),
            (32, 22),
            (35, 0),
            (35, 2),
            (35, 3),
            (35, 7),
            (35, 8),
            (35, 11),
            (35, 12),
            (35, 13),
            (35, 14),
            (35, 16),
            (35, 17),
            (35, 18),
            (35, 20),
            (35, 21),
            (35, 22),
            (35, 23),
            (35, 25),
            (35, 26),
            (35, 27),
            (35, 29),
            (35, 30),
            (35, 32),
            (35, 36),
            (35, 38),
            (35, 39),
            (37, 20),
            (37, 22),
            (38, 20),
            (38, 22),
            (39, 0),
            (39, 2),
            (39, 18),
            (39, 20),
            (39, 21),
            (39, 22),
            (39, 23),
            (39, 26),
            (39, 29),
            (39, 32),
            (39, 38),
            (39, 39),
        }:
            return 18
        elif key in {
            (0, 5),
            (0, 7),
            (0, 8),
            (0, 12),
            (0, 13),
            (0, 14),
            (0, 16),
            (0, 17),
            (0, 19),
            (0, 27),
            (0, 28),
            (2, 7),
            (2, 8),
            (2, 12),
            (2, 13),
            (2, 14),
            (2, 17),
            (2, 19),
            (2, 27),
            (2, 28),
            (3, 0),
            (3, 14),
            (3, 23),
            (3, 29),
            (3, 32),
            (5, 2),
            (5, 12),
            (5, 14),
            (5, 16),
            (5, 17),
            (5, 23),
            (5, 27),
            (6, 0),
            (6, 32),
            (7, 0),
            (10, 0),
            (10, 23),
            (10, 32),
            (11, 0),
            (12, 0),
            (12, 20),
            (12, 22),
            (12, 32),
            (13, 0),
            (13, 23),
            (13, 32),
            (14, 3),
            (14, 6),
            (14, 7),
            (14, 8),
            (14, 10),
            (14, 12),
            (14, 13),
            (14, 14),
            (14, 16),
            (14, 17),
            (14, 19),
            (14, 25),
            (14, 27),
            (14, 28),
            (14, 35),
            (15, 0),
            (15, 14),
            (15, 22),
            (16, 0),
            (16, 14),
            (16, 22),
            (16, 29),
            (16, 32),
            (17, 0),
            (17, 2),
            (17, 23),
            (17, 27),
            (17, 32),
            (18, 3),
            (18, 5),
            (18, 7),
            (18, 8),
            (18, 10),
            (18, 12),
            (18, 13),
            (18, 14),
            (18, 16),
            (18, 17),
            (18, 19),
            (18, 25),
            (18, 27),
            (18, 28),
            (19, 0),
            (19, 2),
            (19, 3),
            (19, 14),
            (19, 17),
            (19, 20),
            (19, 22),
            (19, 23),
            (19, 27),
            (19, 29),
            (19, 32),
            (19, 35),
            (19, 39),
            (20, 5),
            (20, 19),
            (20, 28),
            (21, 0),
            (22, 0),
            (22, 2),
            (22, 3),
            (22, 5),
            (22, 6),
            (22, 7),
            (22, 8),
            (22, 9),
            (22, 10),
            (22, 11),
            (22, 12),
            (22, 13),
            (22, 14),
            (22, 15),
            (22, 16),
            (22, 17),
            (22, 18),
            (22, 19),
            (22, 23),
            (22, 25),
            (22, 27),
            (22, 28),
            (22, 29),
            (22, 30),
            (22, 32),
            (22, 35),
            (22, 36),
            (22, 37),
            (22, 38),
            (22, 39),
            (23, 0),
            (23, 2),
            (23, 12),
            (23, 14),
            (23, 17),
            (23, 18),
            (23, 23),
            (23, 29),
            (23, 32),
            (24, 0),
            (24, 14),
            (24, 17),
            (24, 27),
            (25, 0),
            (25, 14),
            (25, 23),
            (25, 29),
            (25, 32),
            (26, 0),
            (26, 32),
            (27, 0),
            (27, 22),
            (28, 22),
            (29, 5),
            (29, 8),
            (29, 13),
            (29, 14),
            (29, 27),
            (30, 0),
            (30, 14),
            (30, 17),
            (30, 23),
            (30, 29),
            (30, 32),
            (31, 20),
            (31, 22),
            (32, 0),
            (32, 14),
            (32, 23),
            (32, 32),
            (33, 0),
            (33, 22),
            (35, 5),
            (35, 28),
            (36, 0),
            (36, 14),
            (36, 20),
            (36, 22),
            (36, 32),
            (38, 0),
            (38, 14),
            (39, 3),
            (39, 7),
            (39, 8),
            (39, 10),
            (39, 12),
            (39, 13),
            (39, 14),
            (39, 16),
            (39, 17),
            (39, 19),
            (39, 27),
            (39, 28),
            (39, 36),
        }:
            return 34
        elif key in {
            (0, 33),
            (1, 14),
            (2, 33),
            (3, 12),
            (3, 17),
            (3, 24),
            (3, 33),
            (5, 19),
            (5, 24),
            (5, 28),
            (5, 33),
            (6, 14),
            (7, 14),
            (9, 14),
            (10, 12),
            (10, 14),
            (10, 29),
            (10, 33),
            (11, 14),
            (12, 14),
            (12, 29),
            (12, 33),
            (13, 12),
            (13, 14),
            (13, 33),
            (14, 33),
            (16, 33),
            (17, 3),
            (17, 12),
            (17, 14),
            (17, 16),
            (17, 17),
            (17, 18),
            (17, 19),
            (17, 24),
            (17, 29),
            (17, 33),
            (17, 38),
            (17, 39),
            (18, 24),
            (18, 33),
            (19, 11),
            (19, 12),
            (19, 16),
            (19, 18),
            (19, 19),
            (19, 24),
            (19, 26),
            (19, 31),
            (19, 33),
            (19, 38),
            (20, 33),
            (21, 14),
            (22, 24),
            (22, 33),
            (23, 33),
            (27, 14),
            (28, 14),
            (29, 4),
            (29, 6),
            (29, 7),
            (29, 9),
            (29, 10),
            (29, 11),
            (29, 12),
            (29, 15),
            (29, 16),
            (29, 17),
            (29, 19),
            (29, 24),
            (29, 25),
            (29, 28),
            (29, 30),
            (29, 33),
            (29, 35),
            (30, 12),
            (30, 16),
            (30, 33),
            (31, 0),
            (31, 12),
            (31, 14),
            (31, 23),
            (31, 29),
            (31, 32),
            (31, 33),
            (32, 12),
            (32, 17),
            (32, 29),
            (32, 33),
            (33, 14),
            (33, 33),
            (35, 19),
            (35, 24),
            (35, 33),
            (37, 0),
            (37, 14),
            (37, 29),
            (39, 33),
        }:
            return 9
        elif key in {(25, 31), (36, 31), (38, 31)}:
            return 11
        elif key in {(25, 33), (31, 31)}:
            return 10
        elif key in {(38, 37)}:
            return 33
        return 15

    mlp_2_3_outputs = [
        mlp_2_3(k0, k1) for k0, k1 in zip(attn_2_0_outputs, attn_2_3_outputs)
    ]
    mlp_2_3_output_scores = classifier_weights.loc[
        [("mlp_2_3_outputs", str(v)) for v in mlp_2_3_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_0_output, num_attn_2_0_output):
        key = (num_attn_1_0_output, num_attn_2_0_output)
        return 37

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_2_0_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_0_output, num_attn_2_3_output):
        key = (num_attn_2_0_output, num_attn_2_3_output)
        return 5

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_2_0_outputs, num_attn_2_3_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    # num_mlp_2_2 #################################################
    def num_mlp_2_2(num_attn_0_3_output, num_attn_2_1_output):
        key = (num_attn_0_3_output, num_attn_2_1_output)
        return 14

    num_mlp_2_2_outputs = [
        num_mlp_2_2(k0, k1)
        for k0, k1 in zip(num_attn_0_3_outputs, num_attn_2_1_outputs)
    ]
    num_mlp_2_2_output_scores = classifier_weights.loc[
        [("num_mlp_2_2_outputs", str(v)) for v in num_mlp_2_2_outputs]
    ]

    # num_mlp_2_3 #################################################
    def num_mlp_2_3(num_attn_1_0_output, num_attn_2_2_output):
        key = (num_attn_1_0_output, num_attn_2_2_output)
        return 9

    num_mlp_2_3_outputs = [
        num_mlp_2_3(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_2_2_outputs)
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
            "(",
            ")",
            "}",
            "(",
            "(",
            "(",
            ")",
            ")",
            "}",
            "{",
            "}",
            ")",
            ")",
            "{",
            ")",
            "}",
            "{",
            "(",
            "(",
        ]
    )
)
