import io, os, uuid
from slack_bolt import App
from google.cloud import storage

import simple_txt2img

app = App(
    token=os.environ.get('SLACK_BOT_TOKEN'),
    signing_secret=os.environ.get('SLACK_SIGNING_SECRET')
)
client = storage.Client()
bucket = client.get_bucket('img.hideo54.com')

generation_in_progress = False

@app.command('/hideo')
def response_to_command(ack, respond, command):
    ack(response_type='in_channel')
    global generation_in_progress
    global bucket

    channel_sandbox = os.environ.get('CHANNEL_SANDBOX')
    user_hideo54 = os.environ.get('USER_HIDEO54')

    icon_emoji = ':hideo54_bot:'

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

    channel = command['user_id'] if command['channel_name'] == 'directmessage' else command['channel_id']
    if 'text' in command and (command['text'].startswith('sd ') or command['text'].startswith('wd ')):
        for_waifu = command['text'].startswith('wd')
        username = 'Waifu Diffusion' if for_waifu else 'Stable Diffusion'
        if generation_in_progress:
            text = ':fox_face: 同時に相手にできるのは1人だけだこん :pensive: 2分ほど待つこん :tea:'
            app.client.chat_postMessage(
                channel=channel,
                icon_emoji=icon_emoji,
                text=text,
                blocks=create_blocks_from_text(text),  # type: ignore
                username=username,
            )
            return
        else:
            text = ':fox_face: 承ったこん! 生成がんばるこん :muscle: 2分ほどかかるこん… :tea:'
            app.client.chat_postMessage(
                channel=channel,
                icon_emoji=icon_emoji,
                text=text,
                blocks=create_blocks_from_text(text),  # type: ignore
                username=username,
            )
            try:
                prompt = ' '.join(command['text'].split(' ')[1:])
                generation_in_progress = True
                result = simple_txt2img.generate_image(prompt, for_waifu=for_waifu) # takes a long time
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
                    print(prompt, image_urls)
                    description_text = f':fox_face: 「{prompt}」の画像ができあがったこん :muscle: (seed: {seed})'
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
                    if command['channel_id'] != channel_sandbox:
                        app.client.chat_postMessage(
                            channel=user_hideo54, # type: ignore
                            icon_emoji=icon_emoji,
                            text=result_text,
                            blocks=create_blocks_from_text(result_text),  # type: ignore
                            username=username,
                        )
            except:
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
        icon_emoji=':hideo54:',
        text=':fox_face:',
        username='Kaguya in hideout',
    )

if __name__ == '__main__':
    app.start(
        port=int(os.environ.get('PORT', 54011)),
    )
