import io, math, os, re, traceback, uuid
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from google.cloud import storage

import simple_txt2img

app = App(
    token=os.environ.get('SLACK_BOT_TOKEN'),
    signing_secret=os.environ.get('SLACK_SIGNING_SECRET')
)
client = storage.Client()
bucket = client.get_bucket('img.hideo54.com')

channel_sandbox = os.environ.get('CHANNEL_SANDBOX')
user_hideo54 = os.environ.get('USER_HIDEO54')
user_hideo54_bot = os.environ.get('USER_HIDEO54_BOT')

icon_emoji = ':hideo54_bot:'

generation_in_progress = False

def create_blocks_from_text(text: str):
    return [
        {
            'type': 'context',
            'elements': [
                {
                    'type': 'plain_text',
                    'text': 'Managed by Kaguya in hideout',
                    'emoji': True,
                },
            ],
        },
        {
            'type': 'section',
            'text': {
                'type': 'mrkdwn',
                'text': text,
            },
        },
    ]

@app.event('message')
def response_to_command_message(message):
    if 'text' not in message:
        return
    if message['text'].startswith(f'<@{user_hideo54_bot}>') or message['channel_type'] == 'im':
        global generation_in_progress
        global bucket

        channel = message['channel'] if 'channel' in message else message['user']
        command_match = re.search(r'(sd|wd|hd|tr) (portrait |landscape |\d+:\d+ )?(\d* )?(.+)$', message['text'])
        if command_match:
            model_abbv, submode, n_str, prompt = command_match.groups()
            batch_size = int(n_str.strip()) if n_str else 1
            model = 'stable-diffusion'
            if model_abbv == 'wd':
                model = 'waifu-diffusion'
            if model_abbv == 'hd':
                model = 'hentai-diffusion'
            if model_abbv == 'tr':
                model = 'trinart'
            username = model.replace('-', ' ').title()
            estimated_minutes = math.ceil(1.5 + 0.5 * batch_size)
            if generation_in_progress:
                text = f':fox_face: 同時に相手にできるのは1人だけだこん :pensive: {estimated_minutes}分ほど待つこん :tea:'
                app.client.chat_postMessage(
                    channel=channel,
                    icon_emoji=icon_emoji,
                    text=text,
                    blocks=create_blocks_from_text(text),  # type: ignore
                    username=username,
                )
                return
            elif batch_size > 5:
                text = f':fox_face: この bot は欲張りさんの使用を禁止していますこん :blush: :anger:'
                app.client.chat_postMessage(
                    channel=channel,
                    icon_emoji=icon_emoji,
                    text=text,
                    blocks=create_blocks_from_text(text),  # type: ignore
                    username=username,
                )
                return
            else:
                text = f':fox_face: 承ったこん! 生成がんばるこん :muscle: {estimated_minutes}分ほどかかるこん… :tea:'
                app.client.chat_postMessage(
                    channel=channel,
                    icon_emoji=icon_emoji,
                    text=text,
                    blocks=create_blocks_from_text(text),  # type: ignore
                    username=username,
                )
                try:
                    generation_in_progress = True
                    H = 512
                    W = 512
                    if submode is not None:
                        if submode.strip() == 'landscape':
                            W = 640
                            H = 384
                        elif submode.strip() == 'portrait':
                            W = 384
                            H = 640
                        else:
                            w_rate, h_rate = list(map(int, submode.split(':')))
                            if W * H < 64 * 64 * w_rate * h_rate:
                                scaling = math.sqrt(W * H / (w_rate * h_rate))
                                W = int(w_rate * scaling / 64) * 64
                                H = int(h_rate * scaling / 64) * 64
                            else:
                                scaling = math.floor(math.sqrt(W * H / (w_rate * h_rate)) / 64) * 64
                                W = int(w_rate * scaling)
                                H = int(h_rate * scaling)
                    result = simple_txt2img.generate_image(prompt,
                        batch_size=batch_size,
                        W=W,
                        H=H,
                        model=model,
                    ) # takes a long time
                    if result is not None:
                        images, seed = result
                        image_urls = []
                        for image in images:
                            bio = io.BytesIO()
                            image.save(bio, format='png')
                            id = uuid.uuid4()
                            filename = f'stable-diffusion/{id}.png'
                            bucket.blob(filename).upload_from_string(data=bio.getvalue(), content_type='image/png')
                            image_urls.append(f'https://img.hideo54.com/stable-diffusion/{id}.png')
                        print(channel, message['text'], image_urls)
                        seed_text = str(seed) if batch_size == 1 else f'{seed} - {seed + batch_size - 1}'
                        description_text = f':fox_face: 「{prompt}」の画像ができあがったこん :muscle: (seed: {seed_text})'
                        result_text = description_text + '\n' + '\n'.join(image_urls)
                        first_post_result = app.client.chat_postMessage(
                            channel=channel,
                            icon_emoji=icon_emoji,
                            text=description_text,
                            blocks=create_blocks_from_text(description_text),  # type: ignore
                            username=username,
                        )
                        app.client.chat_postMessage(
                            channel=channel,
                            icon_emoji=icon_emoji,
                            text=result_text,
                            blocks=create_blocks_from_text(result_text),  # type: ignore
                            thread_ts=first_post_result['ts'],
                            username=username,
                        )
                        if channel != channel_sandbox and channel != user_hideo54:
                            revealing_text = f'<@{channel}> executed:' + '\n' + result_text
                            app.client.chat_postMessage(
                                channel=user_hideo54, # type: ignore
                                icon_emoji=icon_emoji,
                                text=revealing_text,
                                blocks=create_blocks_from_text(revealing_text),  # type: ignore
                                username=username,
                            )
                except:
                    traceback.print_exc()
                    failed_text = ':fox_face: なんか失敗したこん… :pensive:'
                    app.client.chat_postMessage(
                        channel=channel,
                        icon_emoji=icon_emoji,
                        text=failed_text,
                        blocks=create_blocks_from_text(failed_text),  # type: ignore
                        username=username,
                    )

                generation_in_progress = False
                return

        app.client.chat_postMessage(
            channel=channel,
            icon_emoji=icon_emoji,
            text=':fox_face:',
            username='Kaguya in hideout',
        )

if __name__ == '__main__':
    handler = SocketModeHandler(app, os.environ['SLACK_APP_TOKEN'])
    handler.start()
