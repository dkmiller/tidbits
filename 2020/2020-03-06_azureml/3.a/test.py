import argparse
import json
import stopit
from azureml.core import Workspace


def main(args, ws):
    service = ws.webservices[args.service_name]
    print(f'Testing {service}')
    print('\n---------- LOGS ----------\n')
    print(service.get_logs())

    # NOQA: E501
    x = [[0,1,8,1,0,0,1,0,0,0,0,0,0,0,12,1,0,0,0.5,0.3,0.610327781,7,1,-1,0,-1,1,1,1,2,1,65,1,0.316227766,0.669556409,0.352136337,3.464101615,0.1,0.8,0.6,1,1,6,3,6,2,9,1,1,1,12,0,1,1,0,0,1],[4,2,5,1,0,0,0,0,1,0,0,0,0,0,5,1,0,0,0.9,0.5,0.771362431,4,1,-1,0,0,11,1,1,0,1,103,1,0.316227766,0.60632002,0.358329457,2.828427125,0.4,0.5,0.4,3,3,8,4,10,2,7,2,0,3,10,0,0,1,1,0,1]]
    print (f'x: {x}')

    input_json = json.dumps({"data": x})

    with stopit.ThreadingTimeout(10) as ctx:
        assert ctx.state == ctx.EXECUTING

        predictions = service.run(input_data = input_json)
        print('\n---------- PREDICTION ----------\n')
        print(predictions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--service_name', default='danmill-safe-driver')

    args = parser.parse_args()
    ws = Workspace.from_config()

    main(args, ws)
