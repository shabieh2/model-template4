import * as pulumi from '@pulumi/pulumi';
import * as awsx from '@pulumi/awsx';
import * as aws from '@pulumi/aws';
import * as k8s from '@pulumi/kubernetes';
import * as kx from '@pulumi/kubernetesx';
import TraefikRoute from './TraefikRoute';
import * as docker from "@pulumi/docker";

//

const config = new pulumi.Config();
const baseStack = new pulumi.StackReference("shabieh2/ml-infra/mlplatform")


const repo = new aws.ecr.Repository("myrepo3");


const imageName = repo.repositoryUrl;
const customImage = "myimage";
const imageVersion = "latest"; // in Part 2 this will be dynamic





// [Registry configuration as shown above ...]

// Build and publish the container image.


const image = new docker.Image(customImage, {
  build: "../app7python",
  imageName: pulumi.interpolate`${imageName}:${imageVersion}`,
});
// Export the base and specific version image name.
export const baseImageName = image.baseImageName;
export const fullImageName2 = image.imageName;

//


const registryInfo = repo.registryId.apply(async id => {
  const credentials = await aws.ecr.getCredentials({ registryId: id });
  const decodedCredentials = Buffer.from(credentials.authorizationToken, "base64").toString();
  const [username, password] = decodedCredentials.split(":");
  if (!password || !username) {
      throw new Error("Invalid credentials");
  }
  return {
      server: credentials.proxyEndpoint,
      username: username,
      password: password,
  };
});

//




const provider = new k8s.Provider('provider', {
  kubeconfig: baseStack.requireOutput('kubeconfig'),
  
})


const podBuilder = new kx.PodBuilder({
  containers: [{
    //image: fullImageName, 130 worked, 132 is testing with url_for and not having home and mlplatform for post
    //now testing 133 with no url_for and mlplatform post
    //image:'865053237857.dkr.ecr.us-west-2.amazonaws.com/basicml:134',
    image: fullImageName2,
    ports: { http: 8000 },
    env: {
      'LISTEN_PORT': '8000',
      'MLFLOW_TRACKING_URI': 'http://ml.mlplatform.click/mlflow',
      'MLFLOW_RUN_ID': config.require('runID'),
      
    }
  }],
  serviceAccountName: baseStack.requireOutput('modelsServiceAccountName'),
});




const deployment = new kx.Deployment('mlplatform-serving2', {
  spec: podBuilder.asDeploymentSpec({ replicas: 1  }) 
}, { provider:provider });

const service = deployment.createService();


// Expose model in Traefik 
new TraefikRoute('mlplatform2', {
  prefix: '/lens-tf-model',
  service:service,
  namespace: 'default',
}, { provider: provider, dependsOn: [service]});

//export const traefik_hn= traefik.getResource('v1/Service', 'default/traefik').status.loadBalancer.ingress[0].hostname
//export const service
//export const TraefikRoute
